# -*- coding: utf-8 -*-
"""
Moodle Reaction Question Generator (with Bulk Upload)
@author: Carlos Fernandez Marcos
"""
import streamlit as st
import xml.etree.ElementTree as ET
import requests
import io
import base64
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd  # Required for bulk upload
from my_component import jsme_editor

# ===================================================================
# 1. MODULE AVAILABILITY CHECKS
# ===================================================================

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

PANDAS_AVAILABLE = True  # Already imported above
NUMPY_AVAILABLE = True if 'np' in globals() else False

# ========================================================================
# 2. MULTILINGUAL TEXTS
# ========================================================================

TEXTS = {
    "es": {
        "title": "Generador de Preguntas de Reacción para Moodle",
        "intro": "Crea preguntas con JSME. Normaliza antes de exportar.",
        "change_language": "Change language to English",
        "tab_manual": "Entrada Manual",
        "tab_bulk": "Carga Masiva",
        "search_title": "Búsqueda por Nombre (NCI CIR)",
        "name_input_label": "Nombre del compuesto:",
        "search_button": "Buscar SMILES",
        "add_to_reactants": "Añadir a Reactivos",
        "add_to_products": "Añadir a Productos",
        "reactants_label": "Reactivos (SMILES, separados por comas):",
        "products_label": "Productos (SMILES, separados por comas):",
        "select_missing": "Selecciona la molécula faltante:",
        "reaction_name_label": "Nombre de la Reacción:",
        "correct_feedback_label": "Retroalimentación Correcta:",
        "incorrect_feedback_label": "Retroalimentación Incorrecta (Opcional):",
        "add_reaction_button": "Añadir Reacción",
        "bulk_info": "### Carga Masiva\nSube un archivo Excel/CSV con columnas:\n"
                     "`R1`, `R2`, ..., `P1`, `P2`, ...,`Missing_Name`, 'Correct_Feedback', 'Incorrect_Feedback', 'Reaction_Name'",
        "upload_file_label": "Selecciona archivo Excel/CSV:",
        "process_bulk_button": "Procesar Archivo",
        "processing_bulk": "Procesando {} filas...",
        "bulk_success": "Completado: {} añadidas, {} fallaron.",
        "bulk_error_read": "Error al leer el archivo.",
        "bulk_error_row": "Fila {} omitida: {}",
        "name_error": "No se encontró SMILES para '{}'.",
        "smiles_empty_error": "Los campos no pueden estar vacíos.",
        "smiles_invalid_error": "SMILES inválido: '{}'.",
        "smiles_found": "SMILES encontrado: {}",
        "reaction_added": "Reacción añadida: {}",
        "added_questions_title": "Preguntas Añadidas",
        "normalize_button": "Normalizar con JSME",
        "download_xml_button": "Descargar XML",
        "clear_all_button": "Borrar Todas",
        "no_questions": "Aún no hay preguntas añadidas.",
        "select_molecule_warning": "Selecciona una molécula faltante de la lista.",
        "xml_error": "Error al generar XML: {}",
        "jsme_success": "ÉXITO: **{} de {}** normalizadas correctamente",
        "jsme_partial": "Parcial: **{} de {}** normalizadas",
        "jsme_error": "Ninguna normalizada",
        "normalize_first": "Primero normaliza con JSME"
    },
    "en": {
        "title": "Moodle Reaction Question Generator",
        "intro": "Create questions with JSME. Normalize before export.",
        "change_language": "Cambiar idioma a Español",
        "tab_manual": "Manual Entry",
        "tab_bulk": "Bulk Upload",
        "search_title": "Search by Name (NCI CIR)",
        "name_input_label": "Compound name:",
        "search_button": "Search SMILES",
        "add_to_reactants": "Add to Reactants",
        "add_to_products": "Add to Products",
        "reactants_label": "Reactants (SMILES, comma-separated):",
        "products_label": "Products (SMILES, comma-separated):",
        "select_missing": "Select missing molecule:",
        "reaction_name_label": "Reaction Name:",
        "correct_feedback_label": "Correct Feedback:",
        "incorrect_feedback_label": "Incorrect Feedback (Optional):",
        "add_reaction_button": "Add Reaction",
        "bulk_info": "### Bulk Upload\nUpload Excel/CSV with columns:\n"
                     "`R1`, `R2`, ..., `P1`, `P2`, ...,`Missing_Name`, 'Correct_Feedback', 'Incorrect_Feedback', 'Reaction_Name'",
        "upload_file_label": "Select Excel/CSV file:",
        "process_bulk_button": "Process File",
        "processing_bulk": "Processing {} rows...",
        "bulk_success": "Complete: {} added, {} failed.",
        "bulk_error_read": "Error reading file.",
        "bulk_error_row": "Row {} skipped: {}",
        "name_error": "Could not find SMILES for '{}'.",
        "smiles_empty_error": "Fields cannot be empty.",
        "smiles_invalid_error": "Invalid SMILES: '{}'.",
        "smiles_found": "SMILES found: {}",
        "reaction_added": "Reaction added: {}",
        "added_questions_title": "Added Questions",
        "normalize_button": "Normalize with JSME",
        "download_xml_button": "Download XML",
        "clear_all_button": "Clear All",
        "no_questions": "No questions added yet.",
        "select_molecule_warning": "Select a missing molecule from the list.",
        "xml_error": "Error generating XML: {}",
        "jsme_success": "SUCCESS: **{} of {}** normalized correctly",
        "jsme_partial": "Partial: **{} of {}** normalized",
        "jsme_error": "None normalized",
        "normalize_first": "First normalize with JSME"
    }
}

# ========================================================================
# 3. SESSION STATE INITIALIZATION
# ========================================================================

if 'lang' not in st.session_state:
    st.session_state.lang = 'es'
if "reaction_questions" not in st.session_state:
    st.session_state.reaction_questions = []
if "search_result" not in st.session_state:
    st.session_state.search_result = None
if "reactants_str" not in st.session_state:
    st.session_state.reactants_str = ""
if "products_str" not in st.session_state:
    st.session_state.products_str = ""
if "jsme_normalized" not in st.session_state:
    st.session_state.jsme_normalized = False
if "show_jsme" not in st.session_state:
    st.session_state.show_jsme = False
if "normalized_smiles" not in st.session_state:
    st.session_state.normalized_smiles = {}
if "search_counter" not in st.session_state:
    st.session_state.search_counter = 0

texts = TEXTS[st.session_state.lang]

# ========================================================================
# 4. HELPER FUNCTIONS
# ========================================================================

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_path = "fonts/DejaVuSans.ttf"
    for candidate in [font_path, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_mol_consistent(mol, fixed_bond_length: float = 25.0, padding: int = 10) -> Image.Image:
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
    img_array = np.array(img)
    mask = np.any(img_array != [255, 255, 255], axis=-1)
    y_coords, x_coords = np.nonzero(mask)
    if len(y_coords) == 0:
        return Image.new('RGB', (50, 50), (255, 255, 255))
    y0 = max(0, y_coords.min() - padding)
    x0 = max(0, x_coords.min() - padding)
    y1 = min(img.height, y_coords.max() + 1 + padding)
    x1 = min(img.width, x_coords.max() + 1 + padding)
    return img.crop((x0, y0, x1, y1))

def draw_reaction(reactants_smiles, products_smiles, unscaled_images,
                  base_symbol_size, max_w_final, max_h_final) -> str:
    symbols = []
    if len(reactants_smiles) > 1:
        symbols.extend([" + "] * (len(reactants_smiles) - 1))
    symbols.append(" → ")
    if len(products_smiles) > 1:
        symbols.extend([" + "] * (len(products_smiles) - 1))
    font_base = get_font(base_symbol_size)
    draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    symbol_widths = [draw_temp.textbbox((0, 0), s, font=font_base)[2] + 10 for s in symbols]
    max_symbol_h = max([draw_temp.textbbox((0, 0), s, font=font_base)[3] for s in symbols] or [0])
    total_mol_w = sum(img.width for img in unscaled_images)
    max_h = max(max(img.height for img in unscaled_images), max_symbol_h)
    symbols_width = sum(symbol_widths)
    total_w = total_mol_w + symbols_width
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
    scaled_font_size = max(10, int(base_symbol_size * scale_factor))
    font_scaled = get_font(scaled_font_size)
    for i, img in enumerate(unscaled_images):
        sf_w, sf_h = int(img.width * scale_factor), int(img.height * scale_factor)
        img_to_paste = img.resize((sf_w or 1, sf_h or 1), Image.Resampling.LANCZOS)
        y_pos = (final_h - sf_h) // 2
        final_img.paste(img_to_paste, (int(x_offset), y_pos))
        x_offset += sf_w
        if i < len(unscaled_images) - 1:
            symbol_text = symbols[symbol_idx]
            bbox = draw.textbbox((0, 0), symbol_text, font=font_scaled)
            text_h = bbox[3] - bbox[1]
            y_pos_s = (final_h - text_h) // 2
            draw.text((int(x_offset + 5), y_pos_s), symbol_text, font=font_scaled, fill=(0, 0, 0))
            x_offset += (bbox[2] - bbox[0]) + int(10 * scale_factor)
            symbol_idx += 1
    return image_to_base64(final_img)

def generate_reaction_image(reactants_smiles, products_smiles, missing_smiles):
    if not Chem or not rdMolDraw2D or not NUMPY_AVAILABLE:
        return None
    all_smiles = reactants_smiles + products_smiles
    try:
        missing_index = all_smiles.index(missing_smiles)
    except ValueError:
        return None
    fixed_bond_length = 25.0
    padding = 10
    max_w_final = 600
    max_h_final = 200
    base_symbol_size = int(fixed_bond_length)
    unscaled_images = []
    for i, s in enumerate(all_smiles):
        if i == missing_index:
            avg_h = sum(img.height for img in unscaled_images) / len(unscaled_images) if unscaled_images else fixed_bond_length * 3
            q_font_size = max(28, min(int(avg_h * 0.6), 120))
            font_q = get_font(q_font_size)
            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            bbox = temp_draw.textbbox((0, 0), "?", font=font_q)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = int(q_font_size * 0.4)
            q_w, q_h = text_w + pad, text_h + pad
            img_q = Image.new('RGB', (q_w, q_h), (255, 255, 255))
            dq = ImageDraw.Draw(img_q)
            dq.text(((q_w - text_w) / 2, (q_h - text_h) / 2), "?", font=font_q, fill=(0, 0, 0))
            unscaled_images.append(img_q)
        else:
            mol = Chem.MolFromSmiles(s)
            if mol:
                img = draw_mol_consistent(mol, fixed_bond_length, padding)
                unscaled_images.append(img)
            else:
                blank = Image.new('RGB', (50, 50), (255, 255, 255))
                unscaled_images.append(blank)
    return draw_reaction(reactants_smiles, products_smiles, unscaled_images, base_symbol_size, max_w_final, max_h_final)

def generate_xml(questions, lang: str) -> bytes:
    quiz = ET.Element('quiz')
    prompt = '<p>Dibuja la molécula faltante:</p>' if lang == 'es' else '<p>Draw the missing molecule:</p>'
    
    for q in questions:
        smiles_to_use = q['missing_smiles']
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        
        # Question name
        name_el = ET.SubElement(question, 'name')
        ET.SubElement(name_el, 'text').text = q['name']
        
        # Question with image
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        
        # Right answer
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        escaped_smiles = smiles_to_use.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
        ET.SubElement(answer, 'text').text = f"match({escaped_smiles})"
        
        # Model answer
        model = ET.SubElement(question, 'modelanswer')
        model.text = q['missing_smiles']

        # Feedback RIGHT answer
        if q['correct_feedback']:
            fb = ET.SubElement(answer, 'feedback', format='html')
            fb_text = ET.SubElement(fb, 'text')
            fb_text.text = f"<![CDATA[ <p>{q['correct_feedback']}</p> ]]>"
        
        # Feedback WRONG answer
        if q['incorrect_feedback']:
            ans_inc = ET.SubElement(question, 'answer', fraction='0', format='moodle_auto_format')
            ET.SubElement(ans_inc, 'text').text = "*"
            fb_inc = ET.SubElement(ans_inc, 'feedback', format='html')
            fb_inc_text = ET.SubElement(fb_inc, 'text')
            fb_inc_text.text = f"<![CDATA[ <p>{q['incorrect_feedback']}</p> ]]>"
            ET.SubElement(ans_inc, 'atomcount').text = "0"
    
    # Generate pretty XML
    xml_str = ET.tostring(quiz, encoding='utf-8', method='xml').decode('utf-8')
    import xml.dom.minidom
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8')

def get_smiles_from_name(name: str) -> str | None:
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        smiles = res.text.strip().split('\n')[0]
        if "ERROR" in smiles or (RDKIT_AVAILABLE and Chem.MolFromSmiles(smiles) is None):
            return None
        return smiles
    except:
        return None

def generate_xml(questions, lang: str) -> bytes:
    quiz = ET.Element('quiz')
    prompt = '<p>Dibuja la molécula faltante:</p>' if lang == 'es' else '<p>Draw the missing molecule:</p>'
    for q in questions:
        smiles_to_use = q['missing_smiles']
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        name_el = ET.SubElement(question, 'name')
        ET.SubElement(name_el, 'text').text = q['name']
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        escaped_smiles = smiles_to_use.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
        ET.SubElement(answer, 'text').text = f"match({escaped_smiles})"
        fb = ET.SubElement(answer, 'feedback', format='html')
        ET.SubElement(fb, 'text').text = q['correct_feedback']
        model = ET.SubElement(question, 'modelanswer')
        model.text = q['missing_smiles']
        if q.get('incorrect_feedback'):
            ans_inc = ET.SubElement(question, 'answer', fraction='0', format='moodle_auto_format')
            ET.SubElement(ans_inc, 'text').text = "*"
            fb_inc = ET.SubElement(ans_inc, 'feedback', format='html')
            ET.SubElement(fb_inc, 'text').text = q['incorrect_feedback']
    xml_str = ET.tostring(quiz, encoding='utf-8').decode('utf-8')
    import xml.dom.minidom
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8')

# ========================================================================
# 5. BULK UPLOAD PROCESSING (WITH FEEDBACK + NAME + SMILES CONVERSION)
# ========================================================================

def name_to_smiles(name: str) -> str | None:
    """Convert compound name to SMILES using NCI CIR or validate if already SMILES."""
    if not name or pd.isna(name):
        return None
    name = str(name).strip()
    if not name:
        return None
    # If already valid SMILES, return it
    if RDKIT_AVAILABLE and Chem.MolFromSmiles(name) is not None:
        return name
    # Otherwise, search NCI
    return get_smiles_from_name(name)

def process_bulk_file(uploaded_file):
    """Extracts data from Excel/CSV, converts names to SMILES and generates questions."""
    if not uploaded_file:
        st.warning("No file uploaded.")
        return

    # --- UI: progress and state (BEFORE try) ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Read file ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Error reading file: {e}")
        return

    # --- Validate required column ---
    if 'Missing_Name' not in df.columns:
        status_text.empty()
        progress_bar.empty()
        st.error("Column `Missing_Name` is required.")
        return

    # --- Counters ---
    added = 0
    failed = 0
    logs = []

    # --- PRINCIPAL LOOP ---
    for idx, row in df.iterrows():
        status_text.text(f"Processing row {idx + 2}...")
        progress_bar.progress((idx + 1) / len(df))

        # === 1. Missing_Name ===
        missing_name_raw = row.get('Missing_Name')
        if pd.isna(missing_name_raw) or str(missing_name_raw).strip() == '':
            failed += 1
            logs.append(f"Row {idx+2}: Missing_Name is empty")
            continue
        missing_name = str(missing_name_raw).strip()

        # === 2. Reaction_Name (optional) ===
        reaction_name_raw = row.get('Reaction_Name')
        reaction_name_str = str(reaction_name_raw).strip() if pd.notna(reaction_name_raw) else ""
        
        if reaction_name_str == "" or reaction_name_str.lower() == "nan":
            reaction_name = missing_name
        else:
            reaction_name = reaction_name_str

        # === 3. Feedbacks ===
        def clean_feedback(value):
            if pd.isna(value):
                return ""
            s = str(value).strip()
            if s.lower() in ["", "nan", "n/a", "none", "<null>"]:
                return ""
            return s

        correct_feedback = clean_feedback(row.get('Correct_Feedback'))
        incorrect_feedback = clean_feedback(row.get('Incorrect_Feedback'))
        
        # === 4. Reactants and Products ===
        raw_reactants = []
        raw_products = []
        for col in df.columns:
            if col not in ['Missing_Name', 'Reaction_Name', 'Correct_Feedback', 'Incorrect_Feedback']:
                val = row.get(col)
                if pd.notna(val):
                    s = str(val).strip()
                    if s:
                        if col.startswith('R'):
                            raw_reactants.append(s)
                        elif col.startswith('P'):
                            raw_products.append(s)

        if not raw_reactants and not raw_products:
            failed += 1
            logs.append(f"Row {idx+2}: No reactants or products")
            continue

        # === 5. Convert to SMILES ===
        reactants_smiles = []
        products_smiles = []
        missing_smiles = None

        for name in raw_reactants:
            smiles = name_to_smiles(name)
            if not smiles:
                failed += 1
                logs.append(f"Row {idx+2}: Reactant '{name}' → not found")
                break
            reactants_smiles.append(smiles)
        else:
            for name in raw_products:
                smiles = name_to_smiles(name)
                if not smiles:
                    failed += 1
                    logs.append(f"Row {idx+2}: Product '{name}' → not found")
                    break
                products_smiles.append(smiles)
            else:
                missing_smiles = name_to_smiles(missing_name)
                if not missing_smiles:
                    failed += 1
                    logs.append(f"Row {idx+2}: Missing '{missing_name}' → not found")
                elif missing_smiles not in (reactants_smiles + products_smiles):
                    failed += 1
                    logs.append(f"Row {idx+2}: Missing not in reaction")
                else:
                    img = generate_reaction_image(reactants_smiles, products_smiles, missing_smiles)
                    if not img:
                        failed += 1
                        logs.append(f"Row {idx+2}: Image failed")
                    else:
                        st.session_state.reaction_questions.append({
                            'name': reaction_name,
                            'missing_smiles': missing_smiles,
                            'img_base64': img,
                            'correct_feedback': correct_feedback,
                            'incorrect_feedback': incorrect_feedback
                        })
                        added += 1

    # --- Finalize ---
    progress_bar.empty()
    status_text.empty()

    if added > 0:
        st.success(f"Completed: {added} added, {failed} failed.")
    if failed > 0:
        with st.expander(f"Failed rows ({failed})"):
            for log in logs:
                st.caption(log)

    st.session_state.jsme_normalized = False
    if added > 0:
        st.rerun()

# ========================================================================
# 6. SEARCH HELPERS
# ========================================================================

def search_compound_wrapper(counter: int):
    name = st.session_state.get(f"search_input_{counter}", "").strip()
    if not name:
        return
    with st.spinner("Searching..."):
        smiles = get_smiles_from_name(name)
    if smiles:
        st.session_state.search_result = smiles
    else:
        st.error(texts["name_error"].format(name))

# ========================================================================
# 7. STREAMLIT UI
# ========================================================================

st.set_page_config(page_title=texts["title"], layout="wide")

# Language Switch
col_lang, _ = st.columns([3, 2])
with col_lang:
    txt_col, btn_col = st.columns([2, 2], vertical_alignment="center")
    with txt_col:
        st.markdown(f"**{texts['change_language']}**", unsafe_allow_html=True)
    with btn_col:
        if st.button("EN" if st.session_state.lang == "es" else "ES"):
            st.session_state.lang = "en" if st.session_state.lang == "es" else "es"
            st.rerun()

st.title(texts["title"])
st.markdown(texts["intro"])
st.markdown("---")

input_col, list_col = st.columns([2, 1])

# === INPUT COLUMN ===
with input_col:
    tab1, tab2 = st.tabs([texts["tab_manual"], texts["tab_bulk"]])

    with tab1:
        st.subheader(texts["search_title"])
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            st.text_input(
                texts["name_input_label"],
                key=f"search_input_{st.session_state.search_counter}",
                label_visibility="collapsed",
                on_change=lambda: search_compound_wrapper(st.session_state.search_counter)
            )
        with col_btn:
            if st.button(texts["search_button"], use_container_width=True):
                search_compound_wrapper(st.session_state.search_counter)

        if st.session_state.get("search_result"):
            st.markdown(
                f"""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; margin: 10px 0;">
                    <strong>{texts['smiles_found'].split(':')[0]}:</strong><br>
                    <code>{st.session_state.search_result}</code>
                </div>
                """, unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button(texts["add_to_reactants"], use_container_width=True, key="add_r"):
                    current = st.session_state.reactants_str
                    st.session_state.reactants_str = f"{current}, {st.session_state.search_result}".strip(", ")
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()
            with c2:
                if st.button(texts["add_to_products"], use_container_width=True, key="add_p"):
                    current = st.session_state.products_str
                    st.session_state.products_str = f"{current}, {st.session_state.search_result}".strip(", ")
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()

        st.markdown("---")
        reactants_str = st.text_area(texts["reactants_label"], value=st.session_state.reactants_str, height=80, placeholder="CC(=O)O, O")
        products_str = st.text_area(texts["products_label"], value=st.session_state.products_str, height=80, placeholder="CO, CO2")
        st.session_state.reactants_str = reactants_str
        st.session_state.products_str = products_str
        reactants_list = [s.strip() for s in reactants_str.split(',') if s.strip()]
        products_list = [s.strip() for s in products_str.split(',') if s.strip()]
        all_molecules = reactants_list + products_list

        missing_index = None
        if all_molecules:
            missing_index = st.selectbox(
                texts["select_missing"],
                range(len(all_molecules)),
                format_func=lambda x: f"{all_molecules[x]} ({'R' if x < len(reactants_list) else 'P'})"
            )
        else:
            st.info(texts["select_molecule_warning"])

        next_number = len(st.session_state.reaction_questions) + 1
        default_name = f"{'Reacción' if st.session_state.lang == 'es' else 'Reaction'} {next_number}"
        reaction_name = st.text_input(texts["reaction_name_label"], value=default_name)
        default_fb = '¡Muy bien!' if st.session_state.lang == 'es' else 'Well done!'
        correct_feedback = st.text_area(texts["correct_feedback_label"], value=default_fb)
        incorrect_feedback = st.text_area(texts["incorrect_feedback_label"], placeholder="Optional")

        st.markdown("---")
        if st.button(texts["add_reaction_button"], type="primary", icon=":material/add_task:", use_container_width=True):
            if not reaction_name or not all_molecules or missing_index is None:
                st.error(texts["smiles_empty_error"])
            else:
                missing_smiles = all_molecules[missing_index]
                try:
                    if RDKIT_AVAILABLE:
                        for s in all_molecules:
                            if Chem.MolFromSmiles(s) is None and s not in ['[H+]', '[OH-]', 'H2O', 'O2', 'N2', 'CO2']:
                                st.error(texts["smiles_invalid_error"].format(s))
                                st.stop()
                    img = generate_reaction_image(reactants_list, products_list, missing_smiles)
                    if not img:
                        st.error("Error generating image.")
                        st.stop()
                    st.session_state.reaction_questions.append({
                        'name': reaction_name,
                        'missing_smiles': missing_smiles,
                        'img_base64': img,
                        'correct_feedback': correct_feedback or default_fb,
                        'incorrect_feedback': incorrect_feedback
                    })
                    st.success(texts["reaction_added"].format(reaction_name))
                    st.session_state.reactants_str = ""
                    st.session_state.products_str = ""
                    st.session_state.jsme_normalized = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.markdown(texts["bulk_info"])
        uploaded = st.file_uploader(texts["upload_file_label"], type=['xlsx', 'csv'], key="bulk_uploader")
        if uploaded:
            st.success(f"File uploaded: {uploaded.name}")
            if st.button(texts["process_bulk_button"], type="primary", use_container_width=True):
                process_bulk_file(uploaded)
        else:
            st.info("Upload a file to process.")

# === OUTPUT COLUMN ===
with list_col:
    st.subheader(texts["added_questions_title"])
    if st.session_state.reaction_questions:
        if not st.session_state.jsme_normalized:
            if st.button(texts["normalize_button"], use_container_width=True, type="secondary"):
                st.session_state.show_jsme = True
                st.rerun()

        if st.session_state.show_jsme:
            st.markdown("### Edit each molecule")
            with st.form(key="jsme_form"):
                total = len(st.session_state.reaction_questions)
                for i, q in enumerate(st.session_state.reaction_questions):
                    st.markdown(f"**{i+1}. {q['name']}**")
                    try:
                        jsme_editor(q['missing_smiles'], key=f"jsme_{i}")
                    except NameError:
                        st.warning("`jsme_editor` not available.")
                if st.form_submit_button("Apply Normalization", type="primary", use_container_width=True):
                    success = 0
                    for i in range(total):
                        val = st.session_state.get(f"jsme_{i}")
                        if val and val.strip():
                            st.session_state.reaction_questions[i]['missing_smiles'] = val.strip()
                            success += 1
                    if success == total:
                        st.success(texts["jsme_success"].format(success, total))
                    elif success > 0:
                        st.warning(texts["jsme_partial"].format(success, total))
                    else:
                        st.error(texts["jsme_error"])
                    st.session_state.jsme_normalized = True
                    st.session_state.show_jsme = False
                    st.rerun()

        if st.session_state.jsme_normalized:
            try:
                xml_data = generate_xml(st.session_state.reaction_questions, st.session_state.lang)
                st.download_button(
                    label=texts["download_xml_button"],
                    data=xml_data,
                    file_name="reaction_questions.xml",
                    mime="application/xml",
                    use_container_width=True,
                    icon=":material/file_save:",
                    type="primary"
                )
            except Exception as e:
                st.error(texts["xml_error"].format(e))
        else:
            st.info(texts["normalize_first"])

        if st.button(texts["clear_all_button"], icon=":material/delete:", use_container_width=True):
            st.session_state.reaction_questions = []
            st.session_state.jsme_normalized = False
            st.session_state.show_jsme = False
            st.rerun()

        st.markdown("---")
        for i, q in enumerate(st.session_state.reaction_questions):
            with st.container():
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{i+1}. {q['name']}**")
                    try:
                        st.image(io.BytesIO(base64.b64decode(q['img_base64'])), width=250)
                    except:
                        st.warning("Image display failed.")
                    st.caption(f"Missing: `{q['missing_smiles']}`")
                with c2:
                    if st.button("🗑️", key=f"del_{i}"):
                        del st.session_state.reaction_questions[i]
                        st.rerun()
                st.markdown("---")
    else:
        st.info(texts["no_questions"])
