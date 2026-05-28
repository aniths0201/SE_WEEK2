LangChain + Google Gemini: Educational Explainer Chain

A lightweight, modern Python application demonstrating the use of **LangChain Expression Language (LCEL)** to build an automated educational explainer. 
The application utilizes Google's **Gemini Flash** model to dynamically adapt explanations of complex topics for varying target audiences (e.g., explaining "Generative AI" to a "5-year-old").

## Features

- **LangChain Expression Language (LCEL):** Implements a clean, declarative layout chaining a `ChatPromptTemplate`, `ChatGoogleGenerativeAI` model, and a `StrOutputParser` using the pipe (`|`) operator.
- **Dynamic Prompt Engineering:** Dynamically injects the target audience and topic variables into system and user instructions.
- **Secure Configuration:** Safely manages sensitive API keys using standard environment isolation via `python-dotenv`.

## Project Structure

```text
.
├── .env                  # Environment variables file (git ignored)
├── README.md             # Project documentation
└── explainer.py          # Main application script
