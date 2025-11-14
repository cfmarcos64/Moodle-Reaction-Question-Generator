# Moodle Reaction Question Generator

This is a Streamlit application designed to generate Moodle-compatible questions for chemical reactions, including embedded images of the reactions. Users can input reactants and products as SMILES strings, select a missing molecule, and generate an XML file for Moodle's question bank. Additionally, it supports searching for canonical SMILES by molecule name using the NCI CIR API and bulk uploading questions via Excel/CSV files.

## Features
- **Input Reactions**: Enter reactants and products as SMILES strings, with automatic canonicalization using RDKit.
- **Missing Molecule Selection**: Choose a molecule to be the missing one in the reaction question.
- **Image Generation**: Automatically generate reaction images with consistent molecule sizing using RDKit and Pillow.
- **SMILES Search**: Search for canonical SMILES by molecule name using the NCI CIR API.
- **Bulk Upload**: Upload Excel/CSV files to process multiple reaction questions at once.
- **Multilingual Support**: Interface available in English and Spanish.
- **Moodle XML Export**: Generate a Moodle-compatible XML file for importing questions into Moodle.

## Requirements
To run this application, you need to install the following Python packages:

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

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run your_script_name.py
   ```

## Usage
1. **Launch the Application**: Run the script with `streamlit run your_script_name.py` to start the web interface.
2. **Select Language**: Choose between English or Spanish using the language buttons.
3. **Input Reaction**:
   - Enter reactants and products as comma-separated SMILES strings (e.g., `C, O` for reactants, `CO` for products).
   - Alternatively, use the molecule name search to find SMILES codes via the NCI CIR API.
4. **Select Missing Molecule**: Choose the molecule that will be missing in the reaction question.
5. **Add Feedback**: Provide optional feedback for correct and incorrect answers.
6. **Generate Questions**:
   - Add individual questions to the list by submitting the form.
   - For bulk processing, upload an Excel/CSV file with columns `Missing_Name`, `R1`, `R2`, ..., `P1`, `P2`, ..., and optional `Correct_Feedback` and `Incorrect_Feedback`.
7. **Download XML**: Once questions are added, download the generated Moodle XML file for import into Moodle.
8. **Manage Questions**: View added questions, delete individual questions, or clear all questions.

## File Format for Bulk Upload
The Excel/CSV file for bulk upload should include:
- **Required Columns**:
  - `Missing_Name`: Name of the missing molecule (e.g., `water`).
  - `R1`, `R2`, ...: Names of reactants.
  - `P1`, `P2`, ...: Names of products.
- **Optional Columns**:
  - `Correct_Feedback`: Custom feedback for correct answers (default: "Well done!").
  - `Incorrect_Feedback`: Custom feedback for incorrect answers.

Example CSV:
```csv
Missing_Name,R1,R2,P1,Correct_Feedback,Incorrect_Feedback
water,H2,O2,H2O,Great job!,Try again.
```

## Notes
- Ensure all required modules (`rdkit-pypi`, `Pillow`, `requests`, `pandas`, `numpy`) are installed to enable full functionality (image generation, SMILES search, and bulk processing).
- The application uses the NCI CIR API for SMILES lookup, which requires an internet connection.
- The generated XML is compatible with Moodle's `pmatchjme` question type, which requires the JSME plugin in Moodle.
- Images are embedded as base64-encoded PNGs in the XML file for seamless Moodle integration.

## License
This project is licensed under the CC BY_NC_SA 4.0 License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub for any bugs, feature requests, or improvements.

## Contact
For questions or support, contact the repository maintainer or open an issue on GitHub.