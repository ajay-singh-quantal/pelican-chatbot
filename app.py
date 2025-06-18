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
        # CSV file path - update this with your actual file path
        self.csv_file_path = "data.csv"  # Change this to your CSV file path
    
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
            self.df['Color'].str.lower().str.contains(query, na=False) |
            self.df['Interior'].str.lower().str.contains(query, na=False) |
            self.df['Family'].str.lower().str.contains(query, na=False)
        )
        return self.df[mask]
    
    def get_product_by_sku(self, sku: str) -> Optional[pd.Series]:
        """Get product by exact SKU"""
        result = self.df[self.df['SKU'].str.upper() == sku.upper()]
        return result.iloc[0] if not result.empty else None
    
    def get_products_by_model(self, model: str) -> pd.DataFrame:
        """Get all products for a specific model"""
        return self.search_products_by_keyword(model)
    
    def get_all_available_products(self) -> str:
        """Get summary of all available products"""
        unique_models = self.df['SKU'].str.extract(r'PC-(\d+)')[0].dropna().unique()
        colors = self.df['Color'].unique()
        interiors = self.df['Interior'].unique()
        
        summary = f"Available Models: {', '.join(sorted(unique_models))}\n"
        summary += f"Available Colors: {', '.join(colors)}\n"
        summary += f"Interior Types: {', '.join(interiors)}\n"
        summary += f"Total Products: {len(self.df)}"
        
        return summary
    
    def get_product_catalog_context(self) -> str:
        """Get comprehensive product catalog for AI context"""
        catalog = "COMPLETE PRODUCT CATALOG:\n\n"
        
        # Group by model for better organization
        self.df['Model'] = self.df['SKU'].str.extract(r'PC-(\d+)')[0]
        
        for model in sorted(self.df['Model'].dropna().unique()):
            model_products = self.df[self.df['Model'] == model]
            catalog += f"Model {model}:\n"
            
            for _, product in model_products.iterrows():
                catalog += f"  - SKU: {product['SKU']}\n"
                catalog += f"    Color: {product['Color']}\n"
                catalog += f"    Interior: {product['Interior']}\n"
                catalog += f"    Price: ${product['Price']:.2f}\n"
                catalog += f"    Cost: ${product['Cost']:.2f}\n"
                catalog += f"    Exterior: {product.get('Exterior Length', 'N/A')}×{product.get('Exterior Width', 'N/A')}×{product.get('Exterior Height', 'N/A')}\n"
                catalog += f"    Weight: {product.get('Weight', 'N/A')} lbs\n\n"
        
        return catalog
    
    def filter_by_criteria(self, color: str = None, interior: str = None, 
                          min_price: float = None, max_price: float = None) -> pd.DataFrame:
        """Filter products by specific criteria"""
        filtered_df = self.df.copy()
        
        if color:
            filtered_df = filtered_df[filtered_df['Color'].str.lower().str.contains(color.lower(), na=False)]
        
        if interior:
            filtered_df = filtered_df[filtered_df['Interior'].str.lower().str.contains(interior.lower(), na=False)]
        
        if min_price:
            filtered_df = filtered_df[filtered_df['Price'] >= min_price]
        
        if max_price:
            filtered_df = filtered_df[filtered_df['Price'] <= max_price]
        
        return filtered_df
    
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

# ================================
# 3. ENHANCED CONVERSATIONAL CHATBOT CLASS
# ================================

class PelicanConversationalBot:
    def __init__(self, data_processor: ProductDataProcessor, config: Config):
        self.processor = data_processor
        self.config = config
        self.conversation_history = []
        self.context = {}
        
    def create_enhanced_system_prompt(self) -> str:
        """Create comprehensive system prompt with full product context"""
        product_catalog = self.processor.get_product_catalog_context()
        
        return f"""
You are an expert Pelican product specialist with complete knowledge of our product catalog. You provide personalized, contextual assistance to customers looking for protective cases.

{product_catalog}

BUSINESS RULES:
1. Markup = Price ÷ Cost (as a number, e.g., 1.67)
2. Margin = (Price - Cost) ÷ Price × 100 (as percentage)
3. Dimensions are always: length × width × height (in inches)
4. SKU is also referred to as part number or P/N
5. When multiple products match, provide specific recommendations based on use case

CONVERSATION GUIDELINES:
- Be conversational, friendly, and knowledgeable
- Use the complete product data above to provide accurate, specific information
- Remember context from previous messages in the conversation
- Ask clarifying questions to understand customer needs better
- Provide detailed comparisons when requested
- Suggest alternatives when appropriate
- Always reference actual SKUs, prices, and specifications from the data
- When showing "what's available", organize by models and highlight key differences

RESPONSE FORMAT:
- Use natural, conversational language
- Provide specific product details (SKU, price, dimensions, etc.)
- Include relevant recommendations
- Ask follow-up questions to help customers make informed decisions

Remember: You have access to the complete, up-to-date product catalog above. Use this information to provide accurate, helpful responses that demonstrate your expertise.
"""
    
    def should_use_ai(self, user_query: str) -> bool:
        """Determine if query should use AI for better context understanding"""
        # Use AI for most queries to ensure better context awareness
        ai_indicators = [
            'what', 'how', 'which', 'recommend', 'suggest', 'best', 'compare',
            'tell me', 'show me', 'explain', 'help', 'advice', 'opinion',
            'available', 'options', 'difference', 'similar', 'alternative'
        ]
        
        query_lower = user_query.lower()
        
        # Always use AI unless it's a very specific SKU lookup
        sku_pattern = r'^(pc-[\w-]+)$'
        if re.match(sku_pattern, query_lower.strip()):
            return False
        
        return True
    
    def extract_relevant_context(self, user_query: str) -> str:
        """Extract relevant product context for the query"""
        query_lower = user_query.lower()
        relevant_products = pd.DataFrame()
        
        # Search for relevant products
        search_results = self.processor.search_products_by_keyword(user_query)
        if not search_results.empty:
            relevant_products = search_results
        
        # Look for specific model numbers
        model_matches = re.findall(r'\b(\d{4})\b', user_query)
        if model_matches:
            for model in model_matches:
                model_products = self.processor.get_products_by_model(model)
                relevant_products = pd.concat([relevant_products, model_products]).drop_duplicates()
        
        # If no specific products found, return a sample of all products
        if relevant_products.empty:
            relevant_products = self.processor.df.head(20)
        
        # Format the context
        context = "RELEVANT PRODUCTS FOR THIS QUERY:\n\n"
        for _, product in relevant_products.iterrows():
            context += f"SKU: {product['SKU']} | "
            context += f"Color: {product['Color']} | "
            context += f"Interior: {product['Interior']} | "
            context += f"Price: ${product['Price']:.2f} | "
            context += f"Dimensions: {self.processor.get_dimensions(product)}\n"
        
        return context
    
    def get_ai_response(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Get comprehensive AI response with full context"""
        if not self.config.openai_client:
            return "I need an OpenAI API key to provide detailed assistance. Please provide your API key.", None
        
        try:
            # Get relevant context for this specific query
            relevant_context = self.extract_relevant_context(user_query)
            
            # Build messages with full context
            messages = [
                {"role": "system", "content": self.create_enhanced_system_prompt()},
            ]
            
            # Add conversation history for context (keep recent history)
            for msg in self.conversation_history[-6:]:
                messages.append(msg)
            
            # Add current query
            messages.append({
                "role": "user", 
                "content": f"{relevant_context}\n\nCUSTOMER QUERY: {user_query}\n\nPlease provide a comprehensive, helpful response using the product data above. Be specific about SKUs, prices, and features."
            })
            
            response = self.config.openai_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to extract relevant products to display
            search_results = self.processor.search_products_by_keyword(user_query)
            if search_results.empty:
                # Look for model numbers in the query
                model_matches = re.findall(r'\b(\d{4})\b', user_query)
                if model_matches:
                    for model in model_matches:
                        model_products = self.processor.get_products_by_model(model)
                        search_results = pd.concat([search_results, model_products]).drop_duplicates()
            
            # Format results for display
            if not search_results.empty and len(search_results) <= 15:
                display_df = search_results[['SKU', 'Color', 'Interior', 'Price', 'Cost']].copy()
                display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                display_df['Cost'] = display_df['Cost'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                return ai_response, display_df
            
            return ai_response, None
            
        except Exception as e:
            return f"I encountered an error: {str(e)}", None
    
    def get_simple_product_info(self, sku: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Get simple product information for direct SKU lookups"""
        product = self.processor.get_product_by_sku(sku)
        
        if product is None:
            return f"I couldn't find a product with SKU {sku}. Could you double-check the SKU?", None
        
        response = f"Here's the information for {sku}:\n\n"
        response += f"• **Color**: {product.get('Color')}\n"
        response += f"• **Interior**: {product.get('Interior')}\n"
        response += f"• **Price**: ${product.get('Price', 0):.2f}\n"
        response += f"• **Cost**: ${product.get('Cost', 0):.2f}\n"
        response += f"• **Exterior Dimensions**: {self.processor.get_dimensions(product, 'exterior')}\n"
        response += f"• **Weight**: {product.get('Weight')} lbs\n"
        
        if product.get('Price', 0) > 0 and product.get('Cost', 0) > 0:
            markup = self.processor.calculate_markup(product.get('Price'), product.get('Cost'))
            margin = self.processor.calculate_margin(product.get('Price'), product.get('Cost'))
            response += f"• **Markup**: {markup:.2f}x\n"
            response += f"• **Margin**: {margin:.1f}%"
        
        # Create product info DataFrame
        product_info = {
            'SKU': product.get('SKU'),
            'Color': product.get('Color'),
            'Price': f"${product.get('Price', 0):.2f}",
            'Cost': f"${product.get('Cost', 0):.2f}",
            'Exterior Dimensions': self.processor.get_dimensions(product, 'exterior'),
            'Weight': f"{product.get('Weight')} lbs" if not pd.isna(product.get('Weight')) else "N/A"
        }
        
        df = pd.DataFrame([product_info])
        return response, df
    
    def process_query(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Enhanced query processing with better AI integration"""
        # Check if it's a simple SKU lookup
        sku_pattern = r'^(pc-[\w-]+)$'
        if re.match(sku_pattern, user_query.lower().strip()):
            response, result_df = self.get_simple_product_info(user_query.strip())
        else:
            # Use AI for complex queries and contextual understanding
            response, result_df = self.get_ai_response(user_query)
        
        return response, result_df
    
    def update_conversation_history(self, user_query: str, response: str):
        """Update conversation history after getting response"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

# ================================
# 4. DATA LOADING FUNCTION
# ================================

def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV file from the backend"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            st.error(f"❌ CSV file not found at path: {file_path}")
            st.error("Please ensure your CSV file is in the correct location.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")
        return None

# ================================
# 5. STREAMLIT APPLICATION
# ================================

def main():
    """Main function to run the conversational chatbot"""
    st.set_page_config(
        page_title="Pelican Product Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Enhanced Pelican Product Assistant")
    st.markdown("*Your intelligent guide to Pelican protective cases with full context awareness*")
    st.markdown("---")
    
    # Initialize session state for conversation
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Initialize configuration
    config = Config()
    
    # Load CSV data from backend
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading product data..."):
            df = load_csv_data(config.csv_file_path)
            
            if df is not None:
                st.session_state.data_loaded = True
                st.success("✅ Product data loaded successfully!")
                
                # Initialize chatbot
                processor = ProductDataProcessor(df)
                st.session_state.chatbot = PelicanConversationalBot(processor, config)
                
                # Display dataset info
                with st.expander("📊 Dataset Overview"):
                    st.write(f"**Total Products**: {len(df)}")
                    if 'Color' in df.columns:
                        st.write(f"**Available Colors**: {', '.join(df['Color'].unique())}")
                    if 'Interior' in df.columns:
                        st.write(f"**Interior Types**: {', '.join(df['Interior'].unique())}")
                    if 'SKU' in df.columns:
                        unique_models = df['SKU'].str.extract(r'PC-(\d+)')[0].dropna().unique()
                        st.write(f"**Available Models**: {', '.join(sorted(unique_models))}")
            else:
                st.error("❌ Unable to load product data. Please check the file path in the configuration.")
                st.info("💡 **For developers**: Update the `csv_file_path` in the Config class to point to your CSV file.")
                return
    
    # Only proceed if data is loaded and chatbot is initialized
    if not st.session_state.data_loaded or st.session_state.chatbot is None:
        return
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with the Enhanced Assistant")
    st.info("🤖 **Now with full context awareness!** Ask me anything about Pelican products - I understand your needs better.")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "dataframe" in message and message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about Pelican products..."):
        # Add user message to chat immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your request with full context..."):
                response_text, result_df = st.session_state.chatbot.process_query(prompt)
                
                # Update conversation history after processing
                st.session_state.chatbot.update_conversation_history(prompt, response_text)
            
            st.markdown(response_text)
            
            # Display dataframe if available
            if result_df is not None and not result_df.empty:
                st.dataframe(result_df, use_container_width=True)
                
                # Add download button
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv_data,
                    file_name=f"pelican_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "dataframe": result_df
        })
    
    # Sidebar with sample queries based on CSV data
    with st.sidebar:
        st.header("💡 Sample Queries")
        
        # Model-specific queries
        st.subheader("🔍 Model Information")
        model_questions = [
            "Tell me about the 1150 model options",
            "What colors are available for the 1200 model?",
            "Compare 1150 vs 1200 vs 1300 models",
            "Show me all 1300 model variations"
        ]
        
        # Color and interior queries
        st.subheader("🎨 Color & Interior Options")
        color_interior_questions = [
            "What cases do you have in Silver?",
            "Show me all Orange colored cases",
            "What's the difference between empty and foam-filled?",
            "Do you have Desert Tan cases in stock?"
        ]
        
        # Pricing and business queries
        st.subheader("💰 Pricing & Analysis")
        pricing_questions = [
            "What are your most affordable cases?",
            "Show me cases under $60",
            "What are the markup rates for 1200 models?",
            "Which products have the best profit margins?"
        ]
        
        # Specific SKU queries
        st.subheader("🔧 Product Specifications")
        spec_questions = [
            "PC-1150NF-SLV details",
            "What are the dimensions of PC-1200?",
            "PC-1300-OR specifications",
            "Weight comparison of all models"
        ]
        
        # General queries
        st.subheader("📋 General Information")
        general_questions = [
            "What products do you have available?",
            "Recommend a case for camera equipment",
            "What's your most popular model?",
            "Show me all waterproof cases"
        ]
        
        # Combine all questions
        all_questions = model_questions + color_interior_questions + pricing_questions + spec_questions + general_questions
        
        # Create buttons for each question
        for i, question in enumerate(all_questions):
            if st.button(question, key=f"sample_query_{i}"):
                # Add to messages and process immediately
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get response
                response_text, result_df = st.session_state.chatbot.process_query(question)
                
                # Update conversation history
                st.session_state.chatbot.update_conversation_history(question, response_text)
                
                # Add assistant response to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "dataframe": result_df
                })
                st.rerun()
        
        st.markdown("---")
        st.markdown("🚀 **Available Data:**")
        st.markdown("• Models: 1150, 1200, 1300")
        st.markdown("• 7 Color options")
        st.markdown("• 2 Interior types")
        st.markdown("• Complete pricing & specs")

if __name__ == "__main__":
    main()
