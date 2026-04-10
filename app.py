import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json
import os

# Set up page configurations
st.set_page_config(page_title="RNN Lyric Generator", layout="centered")

st.title("🎶 Lyric Generator")
st.write("Generate lyrics using a pre-trained Recurrent Neural Network.")

@st.cache_resource()
def load_assets():
    model = tf.keras.models.load_model("lyric_model.keras")
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open("sequence_length.json", "r") as jf:
        seq_data = json.load(jf)
        max_len_sequence = seq_data["max_len_sequence"]
    return model, tokenizer, max_len_sequence

if not os.path.exists("lyric_model.keras") or not os.path.exists("tokenizer.pkl"):
    st.warning("Please train the model first by running `topic_lyric_generator_rnn.py`.")
    st.stop()
else:
    with st.spinner("Loading AI Model and Tokenizer..."):
        try:
            model, tokenizer, max_len_sequence = load_assets()
            reverseWordIndex = {v: k for k, v in tokenizer.word_index.items()}
        except Exception as e:
            st.error(f"Error loading assets: {e}")
            st.stop()

def sample_with_temperature(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-10) / temperature
    probabilities = np.exp(probabilities)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

# User Inputs
topic = st.text_input("Topic / Starting Words", "We love")
next_words = st.slider("Number of words to generate:", min_value=10, max_value=300, value=100)
temperature = st.slider("Creativity (Temperature):", min_value=0.1, max_value=2.0, value=0.8, step=0.1)

if st.button("Generate Lyrics"):
    if not topic.strip():
        st.warning("Please enter a topic or starting phrase.")
    else:
        with st.spinner("Generating..."):
            seed_text = topic
            generated_words = seed_text.split()
            current_seed = seed_text
            
            progress_bar = st.progress(0)
            
            for i in range(next_words):
                token_list = tokenizer.texts_to_sequences([current_seed])[0]
                token_list = pad_sequences([token_list], maxlen=max_len_sequence - 1, padding="pre")
                
                predicted_probs = model.predict(token_list, verbose=0)[0]
                predicted_index = sample_with_temperature(predicted_probs, temperature=temperature)
                
                output_word = reverseWordIndex.get(predicted_index, "")
                if not output_word or output_word == "OOV":
                    continue
                
                generated_words.append(output_word)
                current_seed = " ".join(generated_words[-20:])
                
                progress_bar.progress((i + 1) / next_words)
            
            st.write("### Generated Lyrics:")
            
            formatted_lyrics = ""
            words_per_line = 8
            for i in range(0, len(generated_words), words_per_line):
                formatted_lyrics += " ".join(generated_words[i : i + words_per_line]) + "  \n"
            
            st.markdown(formatted_lyrics)
