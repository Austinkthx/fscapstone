# fscapstone
FS 2025 capstone


# DialoGPT Chatbot

## Introduction

The DialoGPT Chatbot is a conversational AI application designed using Microsoft's DialoGPT-medium model. This project features both a command-line interface (CLI) and a graphical user interface (GUI) built with Tkinter, providing flexibility for users who prefer terminal-based interactions or a user-friendly graphical interface.

This chatbot is particularly useful for:

* Testing conversational AI interactions
* Implementing chatbot functionality into larger applications
* Educational purposes to understand NLP and chatbot development

## Features

* Conversational interactions using DialoGPT-medium
* Command-line and graphical user interface
* Persistent conversation context
* Multi-turn conversation capabilities
* Easy setup and execution

## Technologies

* **Python 3**
* **Torch**: A tensor computation library for machine learning.
* **Transformers** (Hugging Face): Used for accessing DialoGPT.
* **Tkinter**: Python's standard GUI library for the GUI frontend.

## Installation

### Requirements

First, ensure you have Python installed (recommended Python 3.10 or higher).

Install required libraries with pip:

```bash
pip install torch transformers
```

If using the GUI:

* For Windows and most Linux distributions, Tkinter usually comes pre-installed.
* For macOS users, install Tkinter if not already available:

```bash
brew install python-tk
```

### Running the Chatbot

#### Command-Line Interface:

Run the chatbot from the terminal:

```bash
python chatbot.py
```

#### Graphical User Interface:

Launch the GUI frontend:

```bash
python chatbot_gui.py
```

Simply enter your messages and interact with the chatbot directly in the provided interface.

## Development Setup

### Cloning the Repository

Clone the repository using git:

```bash
git clone [https://github.com/your-username/your-chatbot-repo.git](https://github.com/Austinkthx/fscapstone)
cd your-chatbot-repo
```

### Setting Up Virtual Environment (Recommended)

Set up a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate # (Linux/macOS)
venv\Scripts\activate # (Windows)

pip install -r requirements.txt
```

### Initial Build

The initial build primarily involves downloading and caching the DialoGPT model on first run, which may take some time (\~350 MB).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributors

* **Austin Wilson** (Maintainer)

## Project Status

This chatbot is currently in an **Alpha** state. It is stable enough for basic conversational tasks and demonstrations but may have limitations or bugs.

## Optional Sections

### Known Issues

* Response times might be slow on initial model loading.
* GUI may freeze temporarily during response generation if running on systems with limited resources.

### Roadmap

* Improve GUI responsiveness using more advanced asynchronous handling.
* Add session saving/loading to preserve conversation history.
* Extend chatbot capabilities with fine-tuned DialoGPT models or custom-trained datasets.
)

---

Thank you for checking out the DialoGPT Chatbot!


