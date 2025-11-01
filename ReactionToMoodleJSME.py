# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 21:29:17 2025

@author: Carlos Fernandez Marcos
"""

# This is a Streamlit script to generate Moodle questions about chemical reactions
# with embedded images. It allows the user to define a reaction and a missing molecule.

import streamlit as st
try:
    from rdkit import Chem
    import rdkit
    st.success(f"RDKit importado! Versión: {rdkit.__version__}")
except Exception as e:
    st.error(f"Error importando RDKit: {e}")
    
import xml.etree.ElementTree as ET
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 1. Module Availability Check and Imports ---

# Check availability of critical chemistry and data modules
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    rdDepictor = None
    rdMolDraw2D = None
    RDKIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

# The requests and numpy imports are already at the top,
# but we'll check their 'availability' for consistency with the original logic.
# In a standard environment, they're typically available, but we'll maintain the check structure.
NCI_CIR_AVAILABLE = (requests is not None)
NUMPY_AVAILABLE = (np is not None)

# Interface texts in different languages
TEXTS = {
    "es": {
        "title": "Generador de preguntas de reacción para Moodle 🧪",
        "intro": "Introduce los reactivos y productos como cadenas SMILES, separados por comas. Si no conoces el código SMILES, usa el buscador por nombre más abajo. **Los SMILES se canonicalizan automáticamente con RDKit** para asegurar su compatibilidad con Moodle/JSME.",
        "reactants_label": "Reactivos SMILES (ej: C, O)",
        "products_label": "Productos SMILES (ej: CO)",
        "select_missing": "Selecciona la molécula faltante en la reacción:",
        "input_warning": "Introduce los SMILES de los reactivos o productos para seleccionar la molécula faltante.",
        "add_button": "Añadir pregunta a la lista",
        "generating_image": "Generando imagen de la reacción...",
        "image_success": "Imagen de la reacción generada con éxito.",
        "question_added": "Pregunta añadida: {}",
        "error_missing_molecule": "Error: La molécula faltante no está en la lista de Reactivos o productos.",
        "error_image_gen": "Hubo un error al generar la imagen. Asegúrate de que los SMILES son válidos.",
        "unexpected_error": "Ocurrió un error inesperado: {}",
        "added_questions_subtitle": "Preguntas añadidas",
        "clear_button": "Borrar todas las preguntas",
        "download_xml_button": "Descargar XML Moodle",
        "xml_success": "Archivo XML generado y listo para descargar.",
        "xml_error": "Error al generar el archivo XML: {}",
        "no_questions_info": "Aún no se han añadido preguntas.",
        "question_text": "Dibuja la molécula faltante en la siguiente reacción:",
        "module_warning": "Advertencia: Los módulos 'rdkit', 'Pillow', 'requests', 'pandas' o 'numpy' no están instalados. No se podrán generar imágenes, buscar SMILES automáticamente o procesar archivos masivos. Instálalos con: 'pip install rdkit-pypi Pillow requests pandas numpy'",
        "search_label": "Buscar SMILES canónico por nombre de molécula (NCI CIR + RDKit)",
        "search_placeholder": "ej: water",
        "search_button": "Buscar",
        "search_result_label": "Resultado de la búsqueda:",
        "add_to_reactants": "Añadir a reactivos",
        "add_to_products": "Añadir a productos",
        "search_success": "Molécula encontrada: {}",
        "search_error_not_found": "Molécula no encontrada por NCI CIR o la estructura no se pudo canonicalizar. Intenta un nombre diferente.",
        "search_error_api": "Error de conexión con la API de NCI CIR. Intenta de nuevo más tarde.",
        "new_question_button": "Nueva pregunta",
        "delete_button": "Eliminar",
        "result_preview": "Vista previa del resultado:",
        "bulk_upload_title": "Carga Masiva de Preguntas (Excel/CSV)",
        "bulk_upload_info": "Sube un archivo Excel (.xlsx) o CSV con las siguientes columnas. **Todas las moléculas deben indicarse por nombre:**\n\n- **`Missing_Name`**: Nombre de la molécula que debe faltar (ej: water).\n- **`R1`, `R2`, `R3`,...**: Nombres de los Reactivos.\n- **`P1`, `P2`, `P3`,...**: Nombres de los productos.\n\nOpcionalmente:\n- **`Correct_Feedback`**: Feedback para respuesta correcta (default: ¡Bien hecho!).\n- **`Incorrect_Feedback`**: Feedback para respuesta incorrecta.",
        "file_uploader_label": "Selecciona archivo Excel/CSV",
        "process_bulk_button": "Procesar preguntas masivas",
        "processing_bulk": "Procesando {count} filas... Esto puede tardar varios minutos debido a la búsqueda de SMILES en línea.",
        "bulk_success": "Procesamiento masivo completado. Se han añadido {added} preguntas ({skipped} fallaron).",
        "bulk_error_read": "Error al leer el archivo. Asegúrate de que el formato es correcto.",
        "bulk_error_row": "Fila {index} omitida: No se pudo obtener el SMILES de una o más moléculas.",
        "correct_feedback_label": "Feedback para respuesta correcta",
        "incorrect_feedback_label": "Feedback para respuestas incorrectas",
        "correct_feedback_default": "¡Muy bien!",
        "delete_tooltip": "Borrar"
    },
    "en": {
        "title": "Moodle Reaction Question Generator 🧪",
        "intro": "Enter reactants and products as SMILES strings, separated by commas. If you don't know the smiles code, use the name search below. **SMILES are automatically canonicalized with RDKit** to ensure compatibility with Moodle/JSME.",
        "reactants_label": "Reactants SMILES (e.g.: C, O)",
        "products_label": "Products SMILES (e.g.: CO)",
        "select_missing": "Select the missing molecule in the reaction:",
        "input_warning": "Enter reactants or products SMILES to select the missing molecule.",
        "add_button": "Add question to list",
        "generating_image": "Generating reaction image...",
        "image_success": "Reaction image generated successfully.",
        "question_added": "Question added: {}",
        "error_missing_molecule": "Error: The missing molecule is not in the list of reactants or products.",
        "error_image_gen": "There was an error generating the image. Make sure the SMILES are valid.",
        "unexpected_error": "An unexpected error occurred: {}",
        "added_questions_subtitle": "Added questions",
        "clear_button": "Clear all questions",
        "download_xml_button": "Download Moodle XML",
        "xml_success": "XML file generated and ready to download.",
        "xml_error": "Error generating XML file: {}",
        "no_questions_info": "No questions have been added yet.",
        "question_text": "Draw the missing molecule in the following reaction:",
        "module_warning": "Warning: The 'rdkit', 'Pillow', 'requests', 'pandas' or 'numpy' modules are not installed. Images cannot be generated, nor can automatic SMILES search or bulk file processing be performed. Install them with: 'pip install rdkit-pypi Pillow requests pandas numpy'",
        "search_label": "Search canonical SMILES by molecule name (NCI CIR + RDKit)",
        "search_placeholder": "e.g.: water",
        "search_button": "Search",
        "search_result_label": "Search result:",
        "add_to_reactants": "Add to reactants",
        "add_to_products": "Add to products",
        "search_success": "Molecule found: {}",
        "search_error_not_found": "Molecule not found by NCI CIR or the structure could not be canonicalized. Try a different name.",
        "search_error_api": "Error connecting to the NCI CIR API. Please try again later.",
        "new_question_button": "New question",
        "delete_button": "Delete",
        "result_preview": "Result preview:",
        "bulk_upload_title": "Bulk Question Upload (Excel/CSV)",
        "bulk_upload_info": "Upload an Excel (.xlsx) or CSV file with the following columns. **All molecules must be entered by name:**\n\n- **`Missing_Name`**: Name of the molecule that should be missing (e.g., water).\n- **`R1`, `R2`, `R3`,...**: Names of the reactants.\n- **`P1`, `P2`, `P3`,...**: Names of the products.\n\nOptionally:\n- **`Correct_Feedback`**: Feedback for correct answer (default: Well done!).\n- **`Incorrect_Feedback`**: Feedback for incorrect answer.",
        "file_uploader_label": "Select Excel/CSV file",
        "process_bulk_button": "Process Bulk Questions",
        "processing_bulk": "Processing {count} rows... This may take several minutes due to online SMILES lookup.",
        "bulk_success": "Bulk processing complete. {added} questions added ({skipped} failed).",
        "bulk_error_read": "Error reading the file. Ensure the format is correct.",
        "bulk_error_row": "Row {index} skipped: Could not get SMILES for one or more molecules.",
        "correct_feedback_label": "Feedback for correct answer",
        "incorrect_feedback_label": "Feedback for wrong answers",
        "correct_feedback_default": "Well done!",
        "delete_tooltip": "Delete"
    }
}

# Streamlit configuration
st.set_page_config(layout="wide")

# Initialize language and session state
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "reaction_questions" not in st.session_state:
    st.session_state.reaction_questions = []
if "search_result_smiles" not in st.session_state:
    st.session_state.search_result_smiles = None
if "search_result_image" not in st.session_state:
    st.session_state.search_result_image = None
if "correct_feedback_input" not in st.session_state:
    st.session_state.correct_feedback_input = TEXTS[st.session_state.lang]["correct_feedback_default"]

# Get texts in the current language
texts = TEXTS[st.session_state.lang]

# --- 3. Utility Functions ---

def draw_mol_consistent(mol, fixed_bond_length=25.0, padding=10):
    """Draws a molecule consistently, returns a cropped PIL Image."""
    if not mol or not rdMolDraw2D or not NUMPY_AVAILABLE:
        return Image.new('RGB', (50, 50), (255, 255, 255))
    
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    
    opts = rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = max(1, int(fixed_bond_length * 0.1))
    opts.fixedBondLength = fixed_bond_length
    opts.fixedFontSize = int(fixed_bond_length * 0.55)
    opts.padding = 0.1
    opts.addStereoAnnotation = False
    opts.clearBackground = True
    
    large_size = 2048
    drawer = rdMolDraw2D.MolDraw2DCairo(large_size, large_size)
    drawer.SetDrawOptions(opts)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    bio = io.BytesIO(drawer.GetDrawingText())
    img = Image.open(bio).convert('RGB')
    
    # Trim whitespace
    img_array = np.array(img)
    mask = np.any(img_array != [255, 255, 255], axis=-1)
    y_coords, x_coords = np.nonzero(mask)
    
    if len(y_coords) == 0:
        return Image.new('RGB', (50, 50), (255, 255, 255))
        
    y0 = max(0, y_coords.min() - padding)
    x0 = max(0, x_coords.min() - padding)
    y1 = min(img.height, y_coords.max() + 1 + padding)
    x1 = min(img.width, x_coords.max() + 1 + padding)
    
    trimmed = img.crop((x0, y0, x1, y1))
    return trimmed

@st.cache_data
def name_to_smiles(compound_name):
    """Converts a compound name to canonical SMILES using NCI/CIR, returns None on failure."""
    if not NCI_CIR_AVAILABLE or not Chem:
        return None
        
    compound_name = str(compound_name).strip()
    if not compound_name:
        return None
        
    try:
        encoded_name = requests.utils.quote(compound_name)
        url = f"http://cactus.nci.nih.gov/chemical/structure/{encoded_name}/smiles"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            smiles = response.text.strip()
            # Handle multiple SMILES returned (take the first) or error strings
            if "\n" in smiles:
                smiles = smiles.split('\n')[0]
                
            if smiles and not any(err in smiles for err in ["Error", "Server Error", "NOT_FOUND"]):
                # Canonicalize SMILES using RDKit
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToSmiles(mol, canonical=True) if mol else None
        return None
    except:
        return None

def escape_smiles(smiles):
    """Escapes special SMILES characters for Moodle's pmatchjme answer format."""
    return smiles.replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]").replace("\\", "\\\\")

def image_to_base64(img):
    """Converts a PIL Image to a base64 encoded string and mimetype."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_encoded, "image/png"

def generate_molecule_image(smiles):
    """Generates an image of a single molecule from SMILES."""
    if not Chem or not rdMolDraw2D or not NUMPY_AVAILABLE:
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = draw_mol_consistent(mol)
            return image_to_base64(img)
        return None, None
    except Exception:
        return None, None

def generate_reaction_image(reactants_smiles, products_smiles, missing_smiles):
    """Generates a reaction image with the missing compound replaced by a question mark."""
    if not Chem or not rdMolDraw2D or not NUMPY_AVAILABLE:
        return None, None

    all_smiles = reactants_smiles + products_smiles
    try:
        missing_index = all_smiles.index(missing_smiles)
    except ValueError:
        return None, None # Missing SMILES not in reaction

    fixed_bond_length = 25.0
    padding = 10
    max_w_final = 600
    max_h_final = 200
    base_symbol_size = int(fixed_bond_length)
    
    unscaled_images = []
    max_h = 0
    total_mol_w = 0

    # 1. Generate individual molecule/question mark images
    for i, s in enumerate(all_smiles):
        if i == missing_index:
            # Create question mark image
            try:
                q_font_size = int(fixed_bond_length * 1.5)
                font_q = ImageFont.truetype("arial.ttf", q_font_size)
            except IOError:
                font_q = ImageFont.load_default()
                q_font_size = 36

            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            bbox = temp_draw.textbbox((0, 0), "?", font=font_q)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            q_w, q_h = text_w + 20, text_h + 20
            
            img_q = Image.new('RGB', (q_w, q_h), (255, 255, 255))
            dq = ImageDraw.Draw(img_q)
            dq.text(((q_w - text_w) / 2, (q_h - text_h) / 2), "?", font=font_q, fill=(0, 0, 0))
            unscaled_images.append(img_q)
            max_h = max(max_h, q_h)
            total_mol_w += q_w
        else:
            # Generate molecule image
            mol = Chem.MolFromSmiles(s)
            if mol:
                img = draw_mol_consistent(mol, fixed_bond_length, padding)
                unscaled_images.append(img)
                max_h = max(max_h, img.height)
                total_mol_w += img.width
            else:
                unscaled_images.append(Image.new('RGB', (50, 50), (255, 255, 255)))
                max_h = max(max_h, 50)
                total_mol_w += 50
    
    # 2. Calculate symbol (' + ' and ' → ') dimensions
    symbols = []
    if len(reactants_smiles) > 1:
        symbols.extend([" + "] * (len(reactants_smiles) - 1))
    symbols.append(" → ")
    if len(products_smiles) > 1:
        symbols.extend([" + "] * (len(products_smiles) - 1))

    try:
        font_base = ImageFont.truetype("arial.ttf", base_symbol_size)
    except IOError:
        font_base = ImageFont.load_default()

    draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    symbol_widths = [draw_temp.textbbox((0, 0), s, font=font_base)[2] + 10 for s in symbols]
    max_symbol_h = max([draw_temp.textbbox((0, 0), s, font=font_base)[3] for s in symbols] or [0])
    
    symbols_width = sum(symbol_widths)
    max_h = max(max_h, max_symbol_h)
    total_w = total_mol_w + symbols_width

    # 3. Scaling
    scale_factor = 1.0
    if total_w > max_w_final:
        scale_factor = max_w_final / total_w
    
    scaled_h = max_h * scale_factor
    if scaled_h > max_h_final:
        scale_factor = max_h_final / max_h
        
    final_w = int(total_w * scale_factor) + 40
    final_h = int(max_h * scale_factor) + 40
    
    final_img = Image.new('RGB', (final_w, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)
    x_offset = 20
    symbol_idx = 0

    # 4. Compose final image
    scaled_font_size = int(base_symbol_size * scale_factor)
    try:
        font_scaled = ImageFont.truetype("arial.ttf", scaled_font_size)
    except IOError:
        font_scaled = ImageFont.load_default()
        
    for i, img in enumerate(unscaled_images):
        # Paste molecule/question mark
        sf_w, sf_h = int(img.width * scale_factor), int(img.height * scale_factor)
        img_to_paste = img.resize((sf_w or 1, sf_h or 1), Image.Resampling.LANCZOS)
        y_pos = (final_h - sf_h) // 2
        final_img.paste(img_to_paste, (int(x_offset), y_pos))
        x_offset += sf_w
        
        # Draw symbol
        if i < len(unscaled_images) - 1:
            symbol_text = symbols[symbol_idx]
            bbox = draw.textbbox((0, 0), symbol_text, font=font_scaled)
            text_h = bbox[3] - bbox[1]
            y_pos_s = (final_h - text_h) // 2
            draw.text((int(x_offset + 5), y_pos_s), symbol_text, font=font_scaled, fill=(0, 0, 0))
            x_offset += (bbox[2] - bbox[0]) + int(10 * scale_factor)
            symbol_idx += 1
            
    return image_to_base64(final_img)

def generate_multiple_reaction_xml(questions_list, lang):
    """Generates the Moodle XML file content."""
    texts = TEXTS[lang]
    quiz = ET.Element("quiz")
    
    # Markers for CDATA replacement post-serialization
    CDATA_MARKER_BASE = "###REACTION_IMAGE_CDATA_BLOCK_"
    substitution_data = []
    feedback_replacements = []
    
    for i, question_data in enumerate(questions_list):
        # Extract data
        name = question_data["name"]
        missing_smiles = question_data["missing_smiles"]
        img_base64 = question_data["img_base64"]
        img_mimetype = question_data["img_mimetype"]
        correct_feedback = question_data.get("correct_feedback", texts["correct_feedback_default"])
        incorrect_feedback = question_data.get("incorrect_feedback", "")
        
        question = ET.Element("question", type="pmatchjme")
        
        # Question Name
        name_text = ET.SubElement(ET.SubElement(question, "name"), "text")
        name_text.text = name
        
        # Question Text (with image)
        questiontext = ET.SubElement(question, "questiontext", format="html")
        qtext = ET.SubElement(questiontext, "text")
        unique_marker = f"{CDATA_MARKER_BASE}{i}###"
        qtext.text = unique_marker # Placeholder for CDATA block

        html_content = f"{texts['question_text']} <br><br> <img src='data:{img_mimetype};base64,{img_base64}'>"
        substitution_data.append({"marker": unique_marker, "cdata_block": f"<![CDATA[{html_content}]]>"})
        
        # Correct Answer (fraction="100")
        answer_correct = ET.SubElement(question, "answer", fraction="100", format="moodle_auto_format")
        answer_text_correct = ET.SubElement(answer_correct, "text")
        answer_text_correct.text = f"match({escape_smiles(missing_smiles)})"
        
        # Correct Feedback
        feedback_correct = ET.SubElement(ET.SubElement(answer_correct, "feedback", format="html"), "text")
        feedback_correct.text = correct_feedback
        feedback_replacements.append({"from": f"<text>{correct_feedback}</text>", "to": f"<text><![CDATA[{correct_feedback}]]></text>"})
        
        # Model answer field (required for pmatchjme)
        modelanswer = ET.SubElement(question, "modelanswer")
        modelanswer.text = missing_smiles
        
        # Incorrect Answer (fraction="0")
        if incorrect_feedback.strip():
            answer_incorrect = ET.SubElement(question, "answer", fraction="0", format="moodle_auto_format")
            ET.SubElement(answer_incorrect, "text").text = "*" # Any other answer
            
            # Incorrect Feedback
            feedback_incorrect = ET.SubElement(ET.SubElement(answer_incorrect, "feedback", format="html"), "text")
            feedback_incorrect.text = incorrect_feedback
            feedback_replacements.append({"from": f"<text>{incorrect_feedback}</text>", "to": f"<text><![CDATA[{incorrect_feedback}]]></text>"})
            
        quiz.append(question)
        
    # Serialize XML to string
    tree = ET.ElementTree(quiz)
    xml_string = ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
    
    # Replace markers with CDATA blocks (required for Moodle import to work correctly)
    for replacement in feedback_replacements:
        xml_string = xml_string.replace(replacement["from"], replacement["to"])
        
    for item in substitution_data:
        xml_string = xml_string.replace(item["marker"], item["cdata_block"])
        
    final_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
    return final_xml.encode("utf-8")

def process_bulk_file(uploaded_file, lang):
    """Processes an Excel/CSV file to add multiple questions."""
    texts = TEXTS[lang]
    added_count, skipped_count = 0, 0
    status_placeholder = st.empty()
    
    try:
        # Read file
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df = df.fillna('')
        
        if 'Missing_Name' not in df.columns:
            st.error(texts["bulk_error_read"])
            return 0, 0
            
        # Identify R and P columns
        r_cols = [col for col in df.columns if col.startswith('R') and col[1:].isdigit()]
        p_cols = [col for col in df.columns if col.startswith('P') and col[1:].isdigit()]
        total_rows = len(df)
        
        for index, row in df.iterrows():
            status_placeholder.info(f"{texts['processing_bulk'].format(count=total_rows)} - Row {index + 1}/{total_rows}")
            
            missing_name = row['Missing_Name']
            missing_smiles = name_to_smiles(missing_name)
            
            if not missing_smiles:
                st.warning(f"{texts['bulk_error_row'].format(index=index + 1)} - Missing molecule ('{missing_name}') not found.")
                skipped_count += 1
                continue
                
            # Get SMILES for reactants and products
            reactants_names = [row[col] for col in r_cols if row[col]]
            products_names = [row[col] for col in p_cols if row[col]]
            
            reactants_smiles = [name_to_smiles(name) for name in reactants_names]
            products_smiles = [name_to_smiles(name) for name in products_names]
            
            # Filter out failures and check consistency
            valid_reactants_smiles = [s for s in reactants_smiles if s]
            valid_products_smiles = [s for s in products_smiles if s]
            
            if len(valid_reactants_smiles) != len(reactants_names) or len(valid_products_smiles) != len(products_names):
                st.warning(f"{texts['bulk_error_row'].format(index=index + 1)} - Failed to retrieve SMILES for one or more reactants/products.")
                skipped_count += 1
                continue
            
            # Check if the missing compound is part of the reaction
            all_smiles = valid_reactants_smiles + valid_products_smiles
            if missing_smiles not in all_smiles:
                st.warning(f"{texts['bulk_error_row'].format(index=index + 1)} - The missing SMILES is not in the reaction. Check file.")
                skipped_count += 1
                continue

            # Generate image
            img_base64, img_mimetype = generate_reaction_image(valid_reactants_smiles, valid_products_smiles, missing_smiles)
            
            if img_base64:
                reaction_name = f"Reaction (Missing: {missing_name}) - Row {index + 1}"
                correct_feedback = row.get('Correct_Feedback', texts["correct_feedback_default"])
                incorrect_feedback = row.get('Incorrect_Feedback', '')
                
                question_data = {
                    "name": reaction_name,
                    "missing_smiles": missing_smiles,
                    "img_base64": img_base64,
                    "img_mimetype": img_mimetype,
                    "correct_feedback": correct_feedback,
                    "incorrect_feedback": incorrect_feedback
                }
                st.session_state.reaction_questions.append(question_data)
                added_count += 1
            else:
                st.warning(f"{texts['bulk_error_row'].format(index=index + 1)} - Failed to generate the reaction image.")
                skipped_count += 1

        status_placeholder.empty()
        st.success(texts["bulk_success"].format(added=added_count, skipped=skipped_count))
        st.rerun()
        
    except Exception as e:
        status_placeholder.empty()
        st.error(texts["bulk_error_read"])
        return 0, 0

# --- 4. Callback Functions ---

def set_language(lang_code):
    """Sets the application language and reloads default feedback."""
    st.session_state.lang = lang_code
    st.session_state.correct_feedback_input = TEXTS[lang_code]["correct_feedback_default"]
    st.rerun()

def clear_all_questions():
    """Clears all questions from the list."""
    st.session_state.reaction_questions = []

def clear_inputs():
    """Clears the reaction input fields."""
    st.session_state.reactants_input = ""
    st.session_state.products_input = ""
    st.session_state.incorrect_feedback_input = ""

def delete_question(index):
    """Deletes a specific question by index."""
    if 0 <= index < len(st.session_state.reaction_questions):
        st.session_state.reaction_questions.pop(index)
    st.rerun()

def search_molecule():
    """Searches for a molecule name to get its canonical SMILES and image."""
    name = st.session_state.search_name
    if not name:
        return
        
    if not NCI_CIR_AVAILABLE or not RDKIT_AVAILABLE:
        st.error(texts["search_error_api"])
        return

    with st.spinner('Searching and canonicalizing SMILES with NCI CIR...'):
        smiles = name_to_smiles(name)
        
    if smiles:
        st.session_state.search_result_smiles = smiles
        img_base64, _ = generate_molecule_image(smiles)
        st.session_state.search_result_image = img_base64
        st.success(texts["search_success"].format(smiles))
    else:
        st.session_state.search_result_smiles = None
        st.session_state.search_result_image = None
        st.error(texts["search_error_not_found"])

def add_smiles_to_input(target):
    """Adds the search result SMILES to the reactants or products input."""
    smiles = st.session_state.search_result_smiles
    if smiles:
        if target == "reactants":
            current_smiles = st.session_state.get("reactants_input", "")
            st.session_state.reactants_input = f"{current_smiles}, {smiles}" if current_smiles else smiles
        elif target == "products":
            current_smiles = st.session_state.get("products_input", "")
            st.session_state.products_input = f"{current_smiles}, {smiles}" if current_smiles else smiles
            
        st.session_state.search_result_smiles = None
        st.session_state.search_result_image = None
        st.rerun()

# --- 5. Streamlit Interface Layout ---

# Show warning if modules are missing
if not RDKIT_AVAILABLE or not NCI_CIR_AVAILABLE or not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
    st.warning(texts["module_warning"])

## Language Selector
st.markdown("###### Select language / Selecciona tu idioma")
flag_col1, flag_col2 = st.columns([1, 1])
with flag_col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Bandera_de_Espa%C3%B1a.svg", width=50)
    if st.button("ES", key="es_btn"):
        set_language("es")
with flag_col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a5/Flag_of_the_United_Kingdom_%281-2%29.svg", width=60)
    if st.button("EN", key="en_btn"):
        set_language("en")

st.title(texts["title"])

main_col, list_col = st.columns([1, 1])

# --- Main Column (Input, Search, Bulk) ---
with main_col:
    st.markdown(texts["intro"])
    
    ## 5.1 Reaction Input Form
    with st.form("reaction_form"):
        st.subheader(texts["add_button"])
        reactants_str = st.text_input(texts["reactants_label"], key="reactants_input")
        products_str = st.text_input(texts["products_label"], key="products_input")
        
        reactants_list = [s.strip() for s in reactants_str.split(',') if s.strip()]
        products_list = [s.strip() for s in products_str.split(',') if s.strip()]
        complete_list = reactants_list + products_list
        
        missing_index = None
        if complete_list and RDKIT_AVAILABLE:
            missing_index = st.selectbox(texts["select_missing"], options=range(len(complete_list)), format_func=lambda x: complete_list[x], key="missing_mol_select")
        elif not RDKIT_AVAILABLE:
             st.warning(texts["module_warning"])
        else:
            st.info(texts["input_warning"])

        col1, col2 = st.columns(2)
        with col1:
            correct_feedback = st.text_area(texts["correct_feedback_label"], key="correct_feedback_input", height=80)
        with col2:
            incorrect_feedback = st.text_area(texts["incorrect_feedback_label"], value="", key="incorrect_feedback_input", height=80)
            
        submitted = st.form_submit_button(texts["add_button"], type="primary", icon=":material/library_add:")

    if submitted and missing_index is not None and RDKIT_AVAILABLE:
        try:
            missing_smiles = complete_list[missing_index].strip()
            
            # Re-generate image to ensure the question mark is correctly placed.
            with st.spinner(texts["generating_image"]):
                img_base64, img_mimetype = generate_reaction_image(reactants_list, products_list, missing_smiles)
            
            if img_base64:
                st.success(texts["image_success"])
                st.image(io.BytesIO(base64.b64decode(img_base64)), caption="Reaction preview")
                
                question_data = {
                    "name": f"Reaction with {missing_smiles} missing",
                    "missing_smiles": missing_smiles,
                    "img_base64": img_base64,
                    "img_mimetype": img_mimetype,
                    "correct_feedback": correct_feedback,
                    "incorrect_feedback": incorrect_feedback
                }
                st.session_state.reaction_questions.append(question_data)
                st.success(texts["question_added"].format(question_data["name"]))
            else:
                st.error(texts["error_image_gen"])
        except Exception as e:
            st.error(texts["unexpected_error"].format(e))
            
    st.button(texts["new_question_button"], on_click=clear_inputs, icon=":material/add_box:")

    # --- 5.2 Molecule Search Section ---
    st.markdown("---")
    st.subheader(texts["search_label"])
    with st.form("search_form"):
        st.text_input(texts["search_placeholder"], key="search_name", label_visibility="collapsed")
        search_button = st.form_submit_button(texts["search_button"], icon=":material/search:")
        
    if search_button:
        search_molecule()
        
    if st.session_state.search_result_smiles:
        st.markdown(f"**{texts['search_result_label']}** `{st.session_state.search_result_smiles}`")
        if st.session_state.search_result_image:
            st.markdown(f"**{texts['result_preview']}**")
            st.image(io.BytesIO(base64.b64decode(st.session_state.search_result_image)), width=200)
            
        col_add_smiles1, col_add_smiles2 = st.columns(2)
        with col_add_smiles1:
            st.button(texts["add_to_reactants"], on_click=add_smiles_to_input, args=("reactants",))
        with col_add_smiles2:
            st.button(texts["add_to_products"], on_click=add_smiles_to_input, args=("products",))

    # --- 5.3 Bulk Upload Section ---
    st.markdown("---")
    st.subheader(texts["bulk_upload_title"])
    st.markdown(texts["bulk_upload_info"])
    
    uploaded_file = st.file_uploader(texts["file_uploader_label"], type=["xlsx", "csv"], accept_multiple_files=False)
    
    if uploaded_file and PANDAS_AVAILABLE and RDKIT_AVAILABLE and NCI_CIR_AVAILABLE and NUMPY_AVAILABLE:
        if st.button(texts["process_bulk_button"]):
            process_bulk_file(uploaded_file, st.session_state.lang)
    elif uploaded_file:
        st.error(texts["module_warning"])

    # --- 5.4 Download and Clear Buttons (Main Column Bottom) ---
    st.markdown("---")
    if st.session_state.reaction_questions:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button(texts["clear_button"], on_click=clear_all_questions, icon=":material/delete:")
        with btn_col2:
            try:
                xml_bytes = generate_multiple_reaction_xml(st.session_state.reaction_questions, st.session_state.lang)
                st.download_button(
                    label=texts["download_xml_button"],
                    data=xml_bytes,
                    file_name="reaction_moodle.xml",
                    mime="application/xml",
                    type="primary",
                    icon=":material/download:"
                )
            except Exception as e:
                st.error(texts["xml_error"].format(e))

# --- List Column (Added Questions) ---
with list_col:
    if st.session_state.reaction_questions:
        st.subheader(texts["added_questions_subtitle"])
        tooltip_text = texts["delete_tooltip"]
        for i, q in enumerate(st.session_state.reaction_questions):
            item_cols = st.columns([4, 1])
            with item_cols[0]:
                st.write(f"{i+1}. **{q['name']}**")
                st.image(io.BytesIO(base64.b64decode(q['img_base64'])), width=250)
                st.write(f"✅ Correct: {q['correct_feedback']}")
                if q.get("incorrect_feedback", "").strip():
                    st.write(f"❌ Incorrect: {q['incorrect_feedback']}")
            with item_cols[1]:
                st.button("🗑️", help=tooltip_text, key=f"delete_{i}", on_click=delete_question, args=(i,))
            st.markdown("---")
    else:

        st.info(texts["no_questions_info"])

