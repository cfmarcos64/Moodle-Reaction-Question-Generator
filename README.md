# Moodle Reaction Question Generator

This is a Streamlit application designed to generate Moodle-compatible questions for chemical reactions, including embedded images of the reactions. Users can input reactants and products as SMILES strings, select a missing molecule, and generate an XML file for Moodle's question bank. Additionally, it supports searching for canonical SMILES by molecule name using the NCI CIR API and bulk uploading questions via Excel/CSV files.

## ✨ Features
- **Input Reactions**: Enter reactants and products as SMILES strings, with automatic canonicalization using RDKit.
- **Missing Molecule Selection**: Choose a molecule to be the missing one in the reaction question.
- **Image Generation**: Automatically generate reaction images with consistent molecule sizing using RDKit and Pillow.
- **SMILES Search**: Search for canonical SMILES by molecule name using the NCI CIR API.
- **Bulk Upload**: Upload Excel/CSV files to process multiple reaction questions at once.
- **Multilingual Support**: Interface available in English and Spanish.
- **Moodle XML Export**: Generate a Moodle-compatible XML file for importing questions into Moodle.

## 🚀 How to Run the Application
There are two primary ways to access and use this tool:

## Option 1: Use the Public Web Application (Recommended)
The application is deployed publicly in Streamlit Cloud and can be accessed directly through this link:
👉 https://moodle-reaction-question-generator.streamlit.app/

## Option 2: Run Locally (Requires Python and Node.js)
To run the application in local development mode, you must run two processes simultaneously in separate terminals: the Streamlit server (Python) and the frontend component development server (Node/npm).
1. Clone the GitHub Repository and Install Python Dependencies:

git clone https://github.com/cfmarcos64/Moodle-Reaction-Question-Generator
cd [repository-name]
// Install Python dependencies (skip if already done)
pip install -r requirements.txt

2. Run the Component Frontend (TERMINAL 1):
This step starts the component development server on http://localhost:3001. This is necessary for Streamlit to connect to the React component and see live changes.
// Navigate to the frontend directory
cd my_component/frontend
// Install JavaScript dependencies (only the first time)
npm install
// Start the component development server
npm run start

Note: Keep this terminal open and running while using the Streamlit application.

3. Run the Streamlit Application (TERMINAL 2):
Open a second terminal. Navigate back to the project root folder and run the application.
// Go back to the root directory
cd ../..
// Execute the main Streamlit application
streamlit run ReactionToMoodleJSME.py

The Streamlit server will automatically connect to the component development server (Terminal 1).


## Requirements
To run this application locally, you need to install the following Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- `streamlit==1.35.0`
- `rdkit-pypi==2022.9.5`
- `Pillow==10.4.0`
- `requests==2.32.3`
- `pandas==2.2.2`
- `numpy==1.26.4`

## 📁 File Structure
The repository is organized into two main parts: the Streamlit Python application and the custom component frontend. The project directory should contain at least these files:

```markdown
📁 MOODLE-REACTION-QUESTION-GENERATOR
|   LICENSE
|   ReactionToMoodleJSME.py
|   README.md
|   requirements.txt
|   
+---📁 fonts
|       DejaVuSans.ttf
|       
\---📁 my_component
    |   __init__.py
    |   
    +---📁 frontend
    |   |   index.html
    |   |   package-lock.json
    |   |   package.json
    |   |   tsconfig.json
    |   |   vite.config.ts
    |   |   
    |   +---📁 build
    |   |   |   index.html
    |   |   |   
    |   |   \---assets
    |   |           index-DWnvittD.js
    |   |           
    |   \---📁 src
    |           index.tsx
    |           MyComponent.tsx
    |           vite-env.d.ts
    |           
    \---📁 __pycache__
            __init__.cpython-313.pyc
```

## 🛠️ How it works
1. **Launch the Application**
2. **Select Language**: Choose between English or Spanish using the language buttons.
3. **Input Reaction**:
   - Enter reactants and products as comma-separated SMILES strings (e.g., `C, O` for reactants, `CO` for products).
   - Alternatively, use the molecule name search to find SMILES codes via the NCI CIR API.
4. **Select Missing Molecule**: Choose the molecule that will be missing in the reaction question.
5. **Add Feedback**: Provide optional feedback for correct and incorrect answers.
6. **Generate Questions**:
   - Add individual questions to the list by submitting the form clicking "Add Reaction" button.
   - For bulk processing, upload an Excel/CSV file with columns `Missing_Name`, `R1`, `R2`, ..., `P1`, `P2`, ..., and optional `Reaction_Name`, `Correct_Feedback` and `Incorrect_Feedback` and click Process File.
7. Convert to JSME-compatible SMILES:
   - Once questions are added, Normalize with JSME.
   - Apply Normalization
8. **Download XML**: Download the generated Moodle XML file for import into Moodle.
9. **Manage Questions**: View added questions, delete individual questions, or clear all questions.

## File Format for Bulk Upload
The Excel/CSV file for bulk upload should include:
- **Required Columns**:
  - `Missing_Name`: Name of the missing molecule (e.g., `water`).
  - `R1`, `R2`, ...: Names of reactants.
  - `P1`, `P2`, ...: Names of products.
- **Optional Columns**:
  - `Correct_Feedback`: Custom feedback for correct answers (default: "Well done!").
  - `Incorrect_Feedback`: Custom feedback for incorrect answers.
  - `Reaction_Name`: Custom question name to be shown in Moodle. If not included the missing molecule name Will be used as question name.

Example CSV:
```csv
Missing_Name,R1,R2,P1,Correct_Feedback,Incorrect_Feedback,Reaction_Name
water,H2,O2,H2O,Great job!,Try again, Esterification.
```

## Notes
- If running the application locally, nsure all required modules (`rdkit-pypi`, `Pillow`, `requests`, `pandas`, `numpy`) are installed to enable full functionality (image generation, SMILES search, and bulk processing).
- The application uses the NCI CIR API for SMILES lookup, which requires an internet connection.
- The generated XML is compatible with Moodle's `pmatchjme` question type, which requires the JSME plugin in Moodle.
- Images are embedded as base64-encoded PNGs in the XML file for seamless Moodle integration.

## License
This project is licensed under the CC BY_NC_SA 4.0 License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub for any bugs, feature requests, or improvements.

## Contact
For questions or support, contact the repository maintainer or open an issue on GitHub.
