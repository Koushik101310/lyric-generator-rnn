# Refactor Lyric Generator with Streamlit UI

The user wants to stop retraining the model on every run and add a simple UI to insert a topic and generate lyrics.

## User Review Required

> [!IMPORTANT]
> To "not train each time", we must train the model *once* and save it to a file. 
> Since we won't train dynamically per topic anymore, the training script will train on a general subset of songs (e.g., 500-1000 consecutive or random songs). When you enter a "topic" in the UI, the app will use your "topic" as the starting (seed) word(s) to guide the generated lyrics. 

If this sounds good, please approve the plan below!

## Proposed Changes

We will split the monolithic design into two main files:

### Data Preparation and Training
#### [MODIFY] [topic_lyric_generator_rnn.py](file:///c:/Users/dmose/Documents/genai/topic_lyric_generator_rnn.py)
- Remove the `TARGET_TOPIC` filter so the model learns from a general corpus of songs.
- Add code to save the model (`lyric_model.keras`).
- Add code to save the Python tokenizer and sequence length (using `pickle` and `json`).
- Remove the generation code from this script (generation will be handled by the UI).

### New Streamlit App
#### [NEW] [app.py](file:///c:/Users/dmose/Documents/genai/app.py)
- A simple local web app using `streamlit` (which is already installed on your system).
- Loads the saved model (`lyric_model.keras`) and tokenizer only once when the app starts.
- Provides a simple UI (Text Input, Slider for words/temperature, and a Generate button).
- Feeds the user inputted topic into the model to predict the next `N` words.

## Verification Plan

### Automated Tests
- Run `python topic_lyric_generator_rnn.py` once to generate and save the model files. *(Note: this may take a few minutes depending on epochs).*
- Start the UI by running `streamlit run app.py` and verify we can seamlessly prompt topics and receive lyrics consecutively without any retraining penalty.
