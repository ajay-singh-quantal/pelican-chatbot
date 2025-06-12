import pandas as pd
import numpy as np
from openai import OpenAI
import json
import re
import os
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. CONFIGURATION
# ================================

class Config:
    """Configuration class for API keys and settings"""
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        # Try to get API key from environment first
        self.openai_api_key = self._get_api_key()
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1000
        self.temperature = 0.1
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or user input"""
        # First try environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            st.warning("🔑 OpenAI API Key not found in environment variables. Please set it.")
            api_key = st.text_input("Enter your OpenAI API key:", type="password")
            
            if not api_key:
                st.warning("⚠️  No API key provided. Chatbot will run in demo mode.")
                return None
        
        return api_key

# ================================
# 2. DATA PROCESSING CLASS
# ================================

class ProductDataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clean_data()
        st.write(f"📊 Processed {len(self.df)} products successfully")
    
    def clean_data(self):
        """Clean and prepare the dataset"""
        # Convert numeric columns
        numeric_columns = ['Cost', 'Price', 'Exterior Length', 'Exterior Width', 'Exterior Height',
                          'Interior Length', 'Interior Width', 'Interior Height', 'Base height', 'Weight']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert date columns if they exist
        date_columns = ['Last Modified', 'Created', 'Last Cost Update']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Clean string columns
        string_columns = ['SKU', 'Color', 'Shopify Title', 'Vendor', 'Interior', 'Family']
        for col in string_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Fill NaN values appropriately
        self.df['Price'] = self.df['Price'].fillna(0)
        self.df['Cost'] = self.df['Cost'].fillna(0)
    
    def search_products(self, query: str) -> pd.DataFrame:
        """Search products by keyword in SKU or Shopify Title"""
        query = query.lower()
        mask = (
            self.df['SKU'].str.lower().str.contains(query, na=False) |
            self.df['Shopify Title'].str.lower().str.contains(query, na=False)
        )
        return self.df[mask]
    
    def get_product_by_sku(self, sku: str) -> Optional[pd.Series]:
        """Get product by exact SKU"""
        result = self.df[self.df['SKU'].str.upper() == sku.upper()]
        return result.iloc[0] if not result.empty else None
    
    def calculate_markup(self, price: float, cost: float) -> float:
        """Calculate markup: Price / Cost"""
        if cost == 0 or pd.isna(cost) or pd.isna(price):
            return 0
        return price / cost
    
    def calculate_margin(self, price: float, cost: float) -> float:
        """Calculate margin: (Price - Cost) / Price * 100"""
        if price == 0 or pd.isna(price) or pd.isna(cost):
            return 0
        return ((price - cost) / price * 100)
    
    def get_dimensions(self, product: pd.Series, dimension_type: str = 'exterior') -> str:
        """Get dimensions in order: length, width, height"""
        if dimension_type.lower() == 'exterior':
            length = product.get('Exterior Length', 'N/A')
            width = product.get('Exterior Width', 'N/A')
            height = product.get('Exterior Height', 'N/A')
        else:
            length = product.get('Interior Length', 'N/A')
            width = product.get('Interior Width', 'N/A')
            height = product.get('Interior Height', 'N/A')
        
        # Format the dimensions properly
        dims = []
        for dim in [length, width, height]:
            if pd.isna(dim) or dim == 'N/A':
                dims.append('N/A')
            else:
                dims.append(f"{dim:.2f}" if isinstance(dim, (int, float)) else str(dim))
        
        return f"{dims[0]} × {dims[1]} × {dims[2]} inches"
    
    def get_products_by_model(self, model: str) -> pd.DataFrame:
        """Get all products for a specific model (e.g., '1150', '1200')"""
        pattern = f"pelican {model}"
        return self.search_products(pattern)

# ================================
# 3. CHATBOT CLASS
# ================================

class PelicanChatbot:
    def __init__(self, data_processor: ProductDataProcessor, config: Config):
        self.processor = data_processor
        self.config = config
        self.conversation_history = []
    
    def create_system_prompt(self) -> str:
        """Create the system prompt with business rules"""
        return """
You are a specialized chatbot for analyzing Pelican product data. Follow these rules strictly:

BUSINESS RULES:
1. Markup = Price ÷ Cost (as a number, e.g., 1.67)
2. Margin = (Price - Cost) ÷ Price × 100 (as percentage)
3. Dimensions are always returned as: length × width × height (in inches)
4. SKU is also called part number or P/N
5. For keyword searches, use Shopify Title field
6. If multiple SKUs match, ask for color/interior specification

RESPONSE FORMAT:
- Numerical values with units (dimensions in inches, cost/price in dollars)
- Use exact terms from dataset for categories/status
- If data is missing, state "Information unavailable"
- If unclear, ask for clarification
- Be precise and data-driven

AVAILABLE DATA COLUMNS:
SKU, Color, Cost, Price, Exterior Length, Exterior Width, Exterior Height, 
Interior Length, Interior Width, Interior Height, Base height, Weight, 
Shopify Title, Vendor, Interior, Family, Last Cost Update

Always base responses strictly on the provided data and be conversational but professional.
"""
    
    def extract_query_intent(self, user_query: str) -> Dict[str, Any]:
        """Extract intent and parameters from user query"""
        query_lower = user_query.lower()
        
        intent_info = {
            'type': 'general',
            'sku': None,
            'model_search': None,
            'keywords': [],
            'requested_info': [],
            'comparison': False
        }
        
        # Extract SKU patterns
        sku_patterns = [
            r'(pc-[\w-]+)',
            r'p[\/\s]*n[\s:]*([a-zA-Z0-9-]+)',
            r'part number[\s:]*([a-zA-Z0-9-]+)',
            r'sku[\s:]*([a-zA-Z0-9-]+)'
        ]
        
        for pattern in sku_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                intent_info['sku'] = matches[0].upper()
                break
        
        # Extract model searches (e.g., "Pelican 1150", "1200")
        model_patterns = [
            r'pelican\s+(\d+\w*)',
            r'(?:^|\s)(\d{4})(?:\s|$)',  # 4-digit model numbers
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                intent_info['model_search'] = matches[0]
                break
        
        # Determine requested information
        info_keywords = {
            'cost_price': ['cost', 'price', 'pricing', 'how much'],
            'dimensions': ['dimension', 'size', 'measurement', 'length', 'width', 'height'],
            'markup': ['markup', 'mark up'],
            'margin': ['margin'],
            'weight': ['weight', 'heavy', 'weigh'],
            'specs': ['spec', 'specification', 'details']
        }
        
        for info_type, keywords in info_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent_info['requested_info'].append(info_type)
        
        # Check for comparison
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
            intent_info['comparison'] = True
        
        return intent_info
    
    def create_context(self, user_query: str, intent: Dict[str, Any]) -> str:
        """Create relevant data context for the query"""
        context_parts = []
        
        # If specific SKU mentioned
        if intent['sku']:
            product = self.processor.get_product_by_sku(intent['sku'])
            if product is not None:
                context_parts.append("PRODUCT FOUND:")
                context_parts.append(self.format_product_details_for_context(product))
            else:
                context_parts.append(f"SKU {intent['sku']} not found in database.")
        
        # If model search
        elif intent['model_search']:
            products = self.processor.get_products_by_model(intent['model_search'])
            if not products.empty:
                context_parts.append(f"PRODUCTS FOR PELICAN {intent['model_search']}:")
                for _, product in products.head(5).iterrows():
                    context_parts.append(f"- {product['SKU']} ({product['Color']}) - ${product.get('Price', 'N/A')}")
            else:
                context_parts.append(f"No products found for Pelican {intent['model_search']}")
        
        # Add dataset summary
        context_parts.append(f"\nDATASET INFO: {len(self.processor.df)} total products")
        
        return "\n".join(context_parts)
    
    def format_product_details_for_context(self, product: pd.Series) -> str:
        """Format product details for AI context"""
        details = []
        details.append(f"SKU: {product.get('SKU', 'N/A')}")
        details.append(f"Color: {product.get('Color', 'N/A')}")
        details.append(f"Cost: ${product.get('Cost', 'N/A')}")
        details.append(f"Price: ${product.get('Price', 'N/A')}")
        details.append(f"Exterior: {self.processor.get_dimensions(product, 'exterior')}")
        details.append(f"Interior: {self.processor.get_dimensions(product, 'interior')}")
        details.append(f"Weight: {product.get('Weight', 'N/A')} lbs")
        details.append(f"Interior Type: {product.get('Interior', 'N/A')}")
        return "\n".join(details)

    def process_query(self, user_query: str) -> pd.DataFrame:
        """Process user query and return response in tabular format"""
        try:
            # Check if OpenAI client is available
            if not self.config.openai_client:
                return self.process_query_local(user_query)
            
            # Extract intent
            intent = self.extract_query_intent(user_query)
            
            # Create context with relevant data
            context = self.create_context(user_query, intent)
            
            # Create messages for OpenAI
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": f"Data Context:\n{context}\n\nUser Query: {user_query}"}
            ]
            
            # Call OpenAI API with new client
            response = self.config.openai_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            bot_response = response.choices[0].message.content
            return self.process_query_local(user_query)  # In case OpenAI is not used
            
        except Exception as e:
            print(f"⚠️  API Error: {str(e)}")
            return self.process_query_local(user_query)
    
    def process_query_local(self, user_query: str) -> pd.DataFrame:
        """Process query locally and return product data as DataFrame"""
        intent = self.extract_query_intent(user_query)
        
        # Handle specific SKU requests
        if intent['sku']:
            product = self.processor.get_product_by_sku(intent['sku'])
            if product is not None:
                product_df = pd.DataFrame([{
                    'SKU': product['SKU'],
                    'Color': product['Color'],
                    'Price': product['Price'],
                    'Cost': product['Cost'],
                    'Markup': self.processor.calculate_markup(product['Price'], product['Cost']),
                    'Margin': self.processor.calculate_margin(product['Price'], product['Cost']),
                    'Dimensions': self.processor.get_dimensions(product)
                }])
                return product_df  # Return a pandas DataFrame
        
        # Handle model searches
        elif intent['model_search']:
            products = self.processor.get_products_by_model(intent['model_search'])
            if not products.empty:
                products_df = pd.DataFrame(products[['SKU', 'Color', 'Price', 'Cost', 'Shopify Title']])
                return products_df  # Return a pandas DataFrame
        
        return pd.DataFrame(columns=["No results found"])

# ================================
# 4. MAIN EXECUTION FOR STREAMLIT
# ================================

def main():
    """Main function to run the chatbot with Streamlit"""
    st.title("🚀 Pelican Product Chatbot v2.0")
    st.markdown("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Load data (example, you can change the file path if needed)
    data_file = "plytix_export-test1-GPT.csv"
    df = pd.read_csv(data_file)
    
    # Initialize components
    processor = ProductDataProcessor(df)
    chatbot = PelicanChatbot(processor, config)
    
    st.write("🤖 Pelican Product Assistant is ready!")
    
    # Chatbot interface
    user_input = st.text_input("🧑 You:", "")
    if user_input:
        product_info = chatbot.process_query(user_input)
        st.write("🤖 Assistant: Here is the product information:")
        st.dataframe(product_info)  # Display in tabular format
        
if __name__ == "__main__":
    main()

