import pandas as pd
import numpy as np
from openai import OpenAI
import json
import re
import os
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. CONFIGURATION
# ================================

class Config:
    """Configuration class for API keys and settings"""
    def __init__(self):
        load_dotenv()
        self.openai_api_key = self._get_api_key()
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1500
        self.temperature = 0.3
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or user input"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            st.warning("🔑 OpenAI API Key not found in environment variables.")
            api_key = st.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
            
            if not api_key:
                st.warning("⚠️ No API key provided. Limited functionality available.")
                return None
        
        return api_key

# ================================
# 2. DATA PROCESSING CLASS
# ================================

class ProductDataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clean_data()
        st.success(f"📊 Processed {len(self.df)} products successfully")
    
    def clean_data(self):
        """Clean and prepare the dataset"""
        # Convert numeric columns
        numeric_columns = ['Cost', 'Price', 'Exterior Length', 'Exterior Width', 'Exterior Height',
                          'Interior Length', 'Interior Width', 'Interior Height', 'Lid height', 
                          'Base height', 'Weight']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Clean string columns
        string_columns = ['SKU', 'Color', 'Vendor', 'Interior', 'Family', 'Label']
        for col in string_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Fill NaN values appropriately
        self.df['Price'] = self.df['Price'].fillna(0)
        self.df['Cost'] = self.df['Cost'].fillna(0)
    
    def search_products_by_keyword(self, query: str) -> pd.DataFrame:
        """Search products by keyword in SKU or Label"""
        query = query.lower()
        mask = (
            self.df['SKU'].str.lower().str.contains(query, na=False) |
            self.df['Label'].str.lower().str.contains(query, na=False, regex=False) |
            self.df['Color'].str.lower().str.contains(query, na=False)
        )
        return self.df[mask]
    
    def get_product_by_sku(self, sku: str) -> Optional[pd.Series]:
        """Get product by exact SKU"""
        result = self.df[self.df['SKU'].str.upper() == sku.upper()]
        return result.iloc[0] if not result.empty else None
    
    def get_products_by_model(self, model: str) -> pd.DataFrame:
        """Get all products for a specific model"""
        pattern = f"{model}"
        return self.search_products_by_keyword(pattern)
    
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
        
        dims = []
        for dim in [length, width, height]:
            if pd.isna(dim) or dim == 'N/A':
                dims.append('N/A')
            else:
                dims.append(f"{dim:.2f}" if isinstance(dim, (int, float)) else str(dim))
        
        return f"{dims[0]} × {dims[1]} × {dims[2]} inches"
    
    def get_product_summary(self, product: pd.Series) -> str:
        """Get a conversational summary of a product"""
        sku = product.get('SKU', 'Unknown')
        color = product.get('Color', 'Unknown')
        price = product.get('Price', 0)
        cost = product.get('Cost', 0)
        interior = product.get('Interior', 'Unknown')
        weight = product.get('Weight', 'Unknown')
        
        summary = f"The {sku} is a {color.lower()} Pelican case"
        
        if interior and interior != 'Unknown':
            summary += f" with {interior.lower()}"
        
        if price > 0:
            summary += f", priced at ${price:.2f}"
        
        if weight != 'Unknown' and not pd.isna(weight):
            summary += f", weighing {weight} lbs"
        
        exterior_dims = self.get_dimensions(product, 'exterior')
        if 'N/A' not in exterior_dims:
            summary += f". Its exterior dimensions are {exterior_dims}"
        
        if cost > 0 and price > 0:
            markup = self.calculate_markup(price, cost)
            margin = self.calculate_margin(price, cost)
            summary += f". The markup is {markup:.2f} and margin is {margin:.1f}%"
        
        return summary + "."

# ================================
# 3. CONVERSATIONAL CHATBOT CLASS
# ================================

class PelicanConversationalBot:
    def __init__(self, data_processor: ProductDataProcessor, config: Config):
        self.processor = data_processor
        self.config = config
        self.conversation_history = []
        self.context = {}
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for conversational AI"""
        return """
You are a friendly and knowledgeable Pelican product assistant. You help customers find information about Pelican protective cases.

BUSINESS RULES:
1. Markup = Price ÷ Cost (as a number, e.g., 1.67)
2. Margin = (Price - Cost) ÷ Price × 100 (as percentage)
3. Dimensions are always returned as: length × width × height (in inches)
4. SKU is also referred to as part number or P/N
5. When multiple products match, ask for clarification about color or interior type

CONVERSATION STYLE:
- Be friendly, helpful, and conversational
- Use natural language, not bullet points or formal lists
- Ask follow-up questions when needed
- Remember context from previous messages
- Explain technical terms when necessary
- Show enthusiasm about helping with product selection

AVAILABLE PRODUCT DATA:
- SKU, Color, Cost, Price, Exterior/Interior Dimensions
- Weight, Interior Type (Empty or Pick and Pluck foam filled)
- Vendor, Family information

Always provide accurate information based on the data and maintain a helpful, conversational tone.
"""
    
    def extract_product_info(self, user_query: str) -> Dict[str, Any]:
        """Extract product information from user query"""
        query_lower = user_query.lower()
        
        info = {
            'skus': [],
            'model_search': None,
            'color': None,
            'interior': None,
            'intent': 'general'
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
                info['skus'].extend([match.upper() for match in matches])
        
        # Extract model numbers
        model_patterns = [
            r'pelican\s+(\d+\w*)',
            r'(?:^|\s)(\d{4})(?:\s|$)',
            r'(\d+)\s*case'
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                info['model_search'] = matches[0]
                break
        
        # Extract colors
        colors = ['black', 'silver', 'orange', 'yellow', 'blue', 'olive drab', 'desert tan']
        for color in colors:
            if color in query_lower:
                info['color'] = color
                break
        
        # Extract interior types
        if any(word in query_lower for word in ['empty', 'no foam']):
            info['interior'] = 'empty'
        elif any(word in query_lower for word in ['foam', 'pick and pluck']):
            info['interior'] = 'pick and pluck foam filled'
        
        # Determine intent
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
            info['intent'] = 'comparison'
        elif any(word in query_lower for word in ['cost', 'price', 'how much', 'expensive']):
            info['intent'] = 'pricing'
        elif any(word in query_lower for word in ['size', 'dimension', 'measurement']):
            info['intent'] = 'dimensions'
        elif any(word in query_lower for word in ['weight', 'heavy']):
            info['intent'] = 'weight'
        elif info['skus'] or info['model_search']:
            info['intent'] = 'product_lookup'
        
        return info
    
    def handle_local_query(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Handle queries using local data processing"""
        info = self.extract_product_info(user_query)
        
        # Handle SKU-specific queries
        if info['skus']:
            if len(info['skus']) == 1:
                return self.handle_single_product(info['skus'][0])
            else:
                return self.handle_comparison_local(info['skus'])
        
        # Handle model searches
        elif info['model_search']:
            return self.handle_model_search(info['model_search'], info['color'], info['interior'])
        
        # Handle general searches
        else:
            return self.handle_general_search(user_query)
    
    def handle_single_product(self, sku: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Handle single product queries"""
        product = self.processor.get_product_by_sku(sku)
        
        if product is None:
            return f"I couldn't find a product with SKU {sku}. Could you double-check the SKU or try searching by model number?", None
        
        summary = self.processor.get_product_summary(product)
        response = f"Here's what I found about the {sku}:\n\n{summary}"
        
        # Create detailed info DataFrame
        product_info = {
            'SKU': product.get('SKU'),
            'Color': product.get('Color'),
            'Price': f"${product.get('Price', 0):.2f}",
            'Cost': f"${product.get('Cost', 0):.2f}",
            'Exterior Dimensions': self.processor.get_dimensions(product, 'exterior'),
            'Interior Dimensions': self.processor.get_dimensions(product, 'interior'),
            'Weight': f"{product.get('Weight')} lbs" if not pd.isna(product.get('Weight')) else "N/A",
            'Interior Type': product.get('Interior')
        }
        
        df = pd.DataFrame([product_info])
        return response, df
    
    def handle_model_search(self, model: str, color: str = None, interior: str = None) -> Tuple[str, Optional[pd.DataFrame]]:
        """Handle model-based searches"""
        products = self.processor.get_products_by_model(model)
        
        if products.empty:
            return f"I couldn't find any products for model {model}. Could you try a different model number?", None
        
        # Filter by color if specified
        if color:
            products = products[products['Color'].str.lower().str.contains(color.lower(), na=False)]
        
        # Filter by interior if specified
        if interior:
            products = products[products['Interior'].str.lower().str.contains(interior.lower(), na=False)]
        
        if products.empty:
            return f"I found the {model} model, but not in the specific color or interior type you mentioned. Let me show you what's available.", None
        
        if len(products) == 1:
            product = products.iloc[0]
            summary = self.processor.get_product_summary(product)
            response = f"Perfect! I found exactly what you're looking for:\n\n{summary}"
        else:
            response = f"I found {len(products)} options for the {model} model. "
            colors = products['Color'].unique()
            interiors = products['Interior'].unique()
            response += f"Available in {', '.join(colors)} colors with {', '.join(interiors)} interior options. "
            response += "Would you like me to show you a specific one or all of them?"
        
        # Create summary DataFrame
        summary_df = products[['SKU', 'Color', 'Interior', 'Price', 'Cost']].copy()
        summary_df['Price'] = summary_df['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        summary_df['Cost'] = summary_df['Cost'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        
        return response, summary_df
    
    def handle_general_search(self, query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Handle general search queries"""
        results = self.processor.search_products_by_keyword(query)
        
        if results.empty:
            return "I couldn't find any products matching your search. Could you try a different term or be more specific about what you're looking for?", None
        
        response = f"I found {len(results)} products that might interest you. "
        if len(results) > 5:
            response += "Here are the most relevant ones:"
            results = results.head(5)
        
        summary_df = results[['SKU', 'Color', 'Interior', 'Price']].copy()
        summary_df['Price'] = summary_df['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        
        return response, summary_df
    
    def handle_comparison_local(self, skus: List[str]) -> Tuple[str, Optional[pd.DataFrame]]:
        """Handle product comparison"""
        if len(skus) < 2:
            return "I need at least two products to compare. Could you provide another SKU?", None
        
        products = []
        for sku in skus[:2]:  # Compare first two
            product = self.processor.get_product_by_sku(sku)
            if product is not None:
                products.append(product)
        
        if len(products) < 2:
            return "I couldn't find one or both of the products you want to compare. Please check the SKUs.", None
        
        response = f"Here's a comparison between {skus[0]} and {skus[1]}:\n\n"
        
        # Create comparison data
        comparison_data = []
        for i, product in enumerate(products):
            comparison_data.append({
                'Product': f"{product['SKU']} ({product['Color']})",
                'Price': f"${product.get('Price', 0):.2f}",
                'Cost': f"${product.get('Cost', 0):.2f}",
                'Exterior Dimensions': self.processor.get_dimensions(product, 'exterior'),
                'Weight': f"{product.get('Weight')} lbs" if not pd.isna(product.get('Weight')) else "N/A",
                'Interior': product.get('Interior')
            })
        
        df = pd.DataFrame(comparison_data)
        return response, df
    
    def get_openai_response(self, user_query: str, context_data: str = "") -> str:
        """Get response from OpenAI when local processing isn't sufficient"""
        if not self.config.openai_client:
            return "I need an OpenAI API key to provide more detailed assistance. Please provide your API key."
        
        try:
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": f"Context data: {context_data}\n\nUser query: {user_query}"}
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-4:]:  # Last 4 messages for context
                messages.append(msg)
            
            response = self.config.openai_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
    
    def process_query(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Main query processing method"""
        # Try local processing first
        local_response, result_df = self.handle_local_query(user_query)
        
        # If local processing found specific data, return it
        if result_df is not None and not result_df.empty:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": local_response})
            return local_response, result_df
        
        # If local processing didn't find data or user needs more help, use OpenAI
        if self.config.openai_client:
            # Get context from search results
            search_results = self.processor.search_products_by_keyword(user_query)
            context_data = ""
            if not search_results.empty:
                context_data = search_results.head(3).to_string()
            
            openai_response = self.get_openai_response(user_query, context_data)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": openai_response})
            
            return openai_response, result_df
        
        # Fallback to local response
        return local_response, result_df

# ================================
# 4. STREAMLIT APPLICATION
# ================================

def main():
    """Main function to run the conversational chatbot"""
    st.set_page_config(
        page_title="Pelican Product Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Pelican Product Conversational Assistant")
    st.markdown("*Your friendly guide to Pelican protective cases*")
    st.markdown("---")
    
    # Initialize session state for conversation
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    
    # Initialize configuration
    config = Config()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV file uploaded successfully!")
    else:
        st.info("📁 Please upload a CSV file to get started.")
        return
    
    # Initialize chatbot if not already done
    if st.session_state.chatbot is None:
        processor = ProductDataProcessor(df)
        st.session_state.chatbot = PelicanConversationalBot(processor, config)
    
    # Display dataset info
    with st.expander("📊 Dataset Overview"):
        st.write(f"**Total Products**: {len(df)}")
        if 'Color' in df.columns:
            st.write(f"**Available Colors**: {', '.join(df['Color'].unique())}")
        if 'Interior' in df.columns:
            st.write(f"**Interior Types**: {', '.join(df['Interior'].unique())}")
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with the Assistant")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "dataframe" in message and message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about Pelican products..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, result_df = st.session_state.chatbot.process_query(prompt)
            
            st.markdown(response_text)
            
            # Display dataframe if available
            if result_df is not None and not result_df.empty:
                st.dataframe(result_df, use_container_width=True)
                
                # Add download button
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv_data,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "dataframe": result_df
        })
    
    # Sidebar with example questions
    with st.sidebar:
        st.header("💡 Try asking:")
        example_questions = [
            "What's the price of PC-1150NF-SLV?",
            "Show me all yellow Pelican cases",
            "Compare PC-1150 and PC-1200",
            "What are the dimensions of the 1300 model?",
            "Which cases have foam interior?",
            "What's the cheapest Pelican case?",
            "Tell me about the silver cases"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                # Simulate user input
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get response
                response_text, result_df = st.session_state.chatbot.process_query(question)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "dataframe": result_df
                })
                st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            if st.session_state.chatbot:
                st.session_state.chatbot.conversation_history = []
            st.rerun()

if __name__ == "__main__":
    main()
