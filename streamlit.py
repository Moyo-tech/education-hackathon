import streamlit as st
import requests
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import easyocr
import pandas as pd

st.set_page_config(page_title="Book Summary App")

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

# Define a function to generate book-related text with GPT-2
def generate_book_text(sentence):
    k = f"Can you give me the summary of this book: , {sentence}"
    input_ids1 = tokenizer.encode(k, return_tensors='pt')
    output1 = model.generate(input_ids1, max_length=5000, num_beams=6, no_repeat_ngram_size=2, early_stopping=True)
    para = tokenizer.decode(output1[0], skip_special_tokens=True)

    y = f"You can find this book {sentence} for free in this website: "
    input_ids2 = tokenizer.encode(y, return_tensors='pt')
    output2 = model.generate(input_ids2, max_length=5000, num_beams=6, no_repeat_ngram_size=2, early_stopping=True)
    web = tokenizer.decode(output2[0], skip_special_tokens=True)

    z = f"These questions are often asked about {sentence} book genre are  "
    input_ids3 = tokenizer.encode(z, return_tensors='pt')
    output3 = model.generate(input_ids3, max_length=500, num_beams=6, no_repeat_ngram_size=2, early_stopping=True)
    question = tokenizer.decode(output3[0], skip_special_tokens=True)
    
    return para, web, question

# Set up the Streamlit app
st.title("Book Summary App")
st.write("Enter an image URL or text to generate a summary of the book.")

input_type = st.selectbox("Select input type:", options=["Image URL", "Text"])

if input_type == "Image URL":
    image_url = st.text_input("Enter image URL:")
    if image_url != "":
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Convert the image to bytes
            image_bytes = requests.get(image_url).content

            # Use EasyOCR to extract text from the uploaded image
            reader = easyocr.Reader(['en'], gpu=True)
            results = reader.readtext(image_bytes)

            # Extract the 'text' column from the dataframe by its position
            df = pd.DataFrame(results, columns=['bbox', 'text', 'conf'])
            text_col = df.iloc[:, 1]
            # Join everything in the text column to form a sentence
            sentence = ' '.join(text_col)

            # Generate book-related text with GPT-2
            para, web, question = generate_book_text(sentence)

            # Display the generated text
            st.header("Book Summary:")
            st.write(para)
            st.header("Website Link:")
            st.write(web)
            st.header("Frequently Asked Questions:")
            st.write(question)

        except Exception as e:
            st.write("Error: ", e)

elif input_type == "Text":
    sentence = st.text_input("Enter text:")
    if sentence != "":
        # Generate book-related text with GPT-2
        para, web, question = generate_book_text(sentence)

        # Display the generated text
        st.header("Book Summary:")
        st.write(para)
        st.header("Website Link:")
        st.write(web)
        st.header("Frequently Asked Questions:")
        st.write(question)


    
