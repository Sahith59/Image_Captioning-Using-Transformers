import streamlit as st
import torch
import os
import tempfile
import io
import base64
from PIL import Image
from transformers import BertTokenizer
from gtts import gTTS
from models import caption
from datasets import coco, utils
from configuration import Config

# Page configuration
st.set_page_config(
    page_title="CATR: Image Captioning",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(version="v3", checkpoint_path=None):
    """Load the CATR model based on version."""
    config = Config()

    if version in ["v1", "v2", "v3"]:
        try:
            # Use trust_repo=True to avoid interactive prompts
            model = torch.hub.load('saahiluppal/catr', version, pretrained=True, trust_repo=True)
            st.success(f"Successfully loaded pretrained {version} model from torch hub.")
        except Exception as e:
            st.error(f"Failed to load model from torch hub: {e}")
            st.info("Building model from scratch...")
            model, _ = caption.build_model(config)
            try:
                # Create model directory
                os.makedirs("checkpoints", exist_ok=True)
                
                # Download a sample model if needed
                sample_model_path = os.path.join("checkpoints", f"{version}_model.pth")
                if not os.path.exists(sample_model_path):
                    st.info(f"Using randomly initialized model for {version}")
            except Exception as download_error:
                st.warning(f"Could not prepare model: {download_error}")
    else:
        model, _ = caption.build_model(config)
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                st.success(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as load_error:
                st.error(f"Error loading checkpoint: {load_error}")
        else:
            st.warning("No checkpoint provided. Using randomly initialized model.")
    
    model = model.to(device)
    model.eval()
    return model, config

@st.cache_resource
def load_tokenizer():
    """Load the BERT tokenizer."""
    return BertTokenizer.from_pretrained('bert-base-uncased')

def create_caption_and_mask(start_token, max_length):
    """Create caption template and attention mask."""
    caption_template = torch.zeros((1, max_length), dtype=torch.long).to(device)
    mask_template = torch.ones((1, max_length), dtype=torch.bool).to(device)
    
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    
    return caption_template, mask_template

def text_to_speech(text):
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Failed to convert text to speech: {e}")
        return None

def predict_caption(model, image, config, tokenizer, beam_size=3):
    """Predict caption for an image using the CATR model."""
    # Transform image
    image = coco.val_transform(image)
    image = image.unsqueeze(0).to(device)
    
    # Get tokens
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    
    # Create caption and mask
    caption, cap_mask = create_caption_and_mask(
        start_token, config.max_position_embeddings
    )
    
    # Generate caption
    with torch.no_grad():
        for i in range(config.max_position_embeddings - 1):
            predictions = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            
            if predicted_id[0] == end_token:
                break
            
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
    
    # Decode caption
    result = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
    return result

def main():
    """Main function for Streamlit app."""
    # Sidebar
    st.sidebar.title("Model Settings")
    model_version = st.sidebar.selectbox(
        "Select model version",
        options=["v3", "v2", "v1"],
        index=0
    )
    
    custom_checkpoint = st.sidebar.checkbox("Use custom checkpoint")
    checkpoint_path = None
    if custom_checkpoint:
        checkpoint_file = st.sidebar.file_uploader("Upload checkpoint file", type=["pth"])
        if checkpoint_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                tmp.write(checkpoint_file.getvalue())
                checkpoint_path = tmp.name
    
    # Load model and tokenizer
    model, config = load_model(model_version, checkpoint_path)
    tokenizer = load_tokenizer()
    
    # Main content
    st.title("📸 CATR: Transformer-based Image Captioning")
    st.markdown("""
    This application uses the CATR (CAption TRansformer) model to generate descriptive captions for images.
    Upload an image and receive a natural language description along with audio playback.
    """)
    
    # Image upload section
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate caption
        with st.spinner("Generating caption..."):
            caption_text = predict_caption(model, image, config, tokenizer)
        
        # Display caption
        with col2:
            st.header("Generated Caption")
            st.subheader(caption_text)
            
            # Convert caption to speech
            with st.spinner("Converting to speech..."):
                audio_bytes = text_to_speech(caption_text)
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                
                # Download button
                b64_audio = base64.b64encode(audio_bytes.read()).decode()
                href = f'<a href="data:audio/mp3;base64,{b64_audio}" download="caption_audio.mp3">Download Audio</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info("""
    **CATR** (CAption TRansformer) is a transformer-based model designed for vision-to-language tasks. 
    
    It was developed by:
    - Sahith Reddy Thummala
    - Manish Yerram
    
    Georgia State University
    """)

if __name__ == "__main__":
    main()