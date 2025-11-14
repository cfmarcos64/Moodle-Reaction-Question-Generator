# -*- coding: utf-8 -*-
"""
Generador de Preguntas de Reacción para Moodle
@author: Carlos Fernandez Marcos
"""
import streamlit as st
import xml.etree.ElementTree as ET
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np # Importación de numpy
from my_component import jsme_editor

# --- 1. Module Availability Check and Imports ---
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

try:
    # Definición de NUMPY_AVAILABLE (el error que faltaba)
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# --- 2. TEXTS (Multilingüe) ---
TEXTS = {
    "es": {
        "title": "Generador de Preguntas de Reacción para Moodle",
        "intro": "Crea preguntas con JSME. Normaliza antes de exportar.",
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
        "bulk_info": "### Carga Masiva\nSube un archivo Excel/CSV con columnas: `Missing_Name`, `R1`, `R2`, ..., `P1`, `P2`, ...",
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
        "delete_tooltip": "Eliminar",
        "no_questions": "Aún no hay preguntas añadidas.",
        "select_molecule_warning": "Selecciona una molécula faltante de la lista.",
        "xml_error": "Error al generar XML: {}",
        "processing_jsme": "Procesando con JSME...",
        "jsme_success": "ÉXITO: **{} de {}** normalizadas correctamente",
        "jsme_partial": "Parcial: **{} de {}** normalizadas",
        "jsme_error": "Ninguna normalizada",
        "normalize_first": "Primero normaliza con JSME"
    },
    "en": {
        "title": "Moodle Reaction Question Generator",
        "intro": "Create questions with JSME. Normalize before export.",
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
        "bulk_info": "### Bulk Upload\nUpload Excel/CSV with columns: `Missing_Name`, `R1`, `R2`, ..., `P1`, `P2`, ...",
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
        "delete_tooltip": "Delete",
        "no_questions": "No questions added yet.",
        "select_molecule_warning": "Select a missing molecule from the list.",
        "xml_error": "Error generating XML: {}",
        "processing_jsme": "Processing with JSME...",
        "jsme_success": "SUCCESS: **{} of {}** normalized correctly",
        "jsme_partial": "Partial: **{} of {}** normalized",
        "jsme_error": "None normalized",
        "normalize_first": "First normalize with JSME"
    }
}

# --- 3. Session State ---
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

texts = TEXTS[st.session_state.lang]

# --- 4. Core Helper Functions ---

# 4.1. Helper to convert PIL Image to base64 (MISSING)
def image_to_base64(img):
    buffered = io.BytesIO()
    # Ensure correct format for base64 encoding
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 4.2. RDKit drawing function that ensures consistent bond length (REPLACING draw_mol_image)
def draw_mol_consistent(mol, fixed_bond_length, padding, width=300, height=300):
    if not mol or not RDKIT_AVAILABLE:
        img = Image.new('RGB', (width, height), (240, 240, 240))
        return img
    
    # 1. Compute 2D coordinates
    rdDepictor.Compute2DCoords(mol)
    
    # 2. Setup drawer with fixed bond length
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.SetFontSize(0.8)
    drawer.SetLineWidth(2.0)
    # Set the crucial fixed bond length for proportional drawing
    drawer.drawOptions().fixedBondLength = fixed_bond_length 
    
    # 3. Draw and get PNG bytes
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    
    # 4. Load into PIL and crop (to remove RDKit's excess white space)
    img_pil = Image.open(io.BytesIO(png_data)).convert("RGB")
    
    # Simple cropping logic (requires numpy for getbbox)
    if img_pil.getbbox():
        cropped_img = img_pil.crop(img_pil.getbbox())
        
        # Add back required padding
        final_w = cropped_img.width + 2 * padding
        final_h = cropped_img.height + 2 * padding
        final_img = Image.new('RGB', (final_w, final_h), (255, 255, 255))
        final_img.paste(cropped_img, (padding, padding))
        return final_img
    
    return img_pil # Fallback

# 4.3. Main image generation function
def generate_reaction_image(reactants_smiles, products_smiles, missing_smiles):
    """Generates a reaction image with the missing compound replaced by a question mark."""
    # The fix for NUMPY_AVAILABLE is applied in Section 1, allowing this check to run
    if not Chem or not rdMolDraw2D or not NUMPY_AVAILABLE:
        # Returning a placeholder base64 string if dependencies are missing
        img = Image.new('RGB', (500, 100), (240, 240, 240))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Error: RDKit/Numpy not available", fill=(0,0,0))
        return image_to_base64(img)

    all_smiles = reactants_smiles + products_smiles
    try:
        missing_index = all_smiles.index(missing_smiles)
    except ValueError:
        return None # Missing SMILES not in reaction

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
                # Use a standard font if available
                q_font_size = int(fixed_bond_length * 1.5)
                font_q = ImageFont.truetype("arial.ttf", q_font_size)
            except IOError:
                # Fallback to default font
                font_q = ImageFont.load_default()
                q_font_size = 36

            temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            # Adjust bbox calculation for consistent font measurement
            bbox = temp_draw.textbbox((0, 0), "?", font=font_q)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            # Add horizontal padding to the question mark box
            q_w, q_h = text_w + 2 * padding, text_h + 2 * padding
            
            img_q = Image.new('RGB', (q_w, q_h), (255, 255, 255))
            dq = ImageDraw.Draw(img_q)
            # Center the question mark within its padded box
            dq.text(((q_w - text_w) / 2, (q_h - text_h) / 2 - 5), "?", font=font_q, fill=(0, 0, 0))
            unscaled_images.append(img_q)
            max_h = max(max_h, q_h)
            total_mol_w += q_w
        else:
            # Generate molecule image using proportional function
            mol = Chem.MolFromSmiles(s)
            if mol:
                # Calling the new proportional drawing function
                img = draw_mol_consistent(mol, fixed_bond_length, padding) 
                unscaled_images.append(img)
                max_h = max(max_h, img.height)
                total_mol_w += img.width
            else:
                # Fallback for invalid SMILES
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
    # Calculate symbol width using the actual font and text, plus padding
    symbol_widths = [draw_temp.textbbox((0, 0), s, font=font_base)[2] - draw_temp.textbbox((0, 0), s, font=font_base)[0] + int(10) for s in symbols]
    max_symbol_h = max([draw_temp.textbbox((0, 0), s, font=font_base)[3] - draw_temp.textbbox((0, 0), s, font=font_base)[1] for s in symbols] or [0])
    
    symbols_width = sum(symbol_widths)
    max_h = max(max_h, max_symbol_h + 2 * padding) # Add padding to max symbol height
    total_w = total_mol_w + symbols_width

    # 3. Scaling
    scale_factor = 1.0
    if total_w > max_w_final:
        scale_factor = max_w_final / total_w
    
    # Check if height needs to limit scaling more aggressively
    scaled_h = max_h * scale_factor
    if scaled_h > max_h_final:
        scale_factor = max_h_final / max_h
        
    final_w = int(total_w * scale_factor) + 40 # Add margin
    final_h = int(max_h * scale_factor) + 40 # Add margin
    
    final_img = Image.new('RGB', (final_w, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)
    x_offset = 20 # Initial margin
    symbol_idx = 0

    # 4. Compose final image
    scaled_font_size = max(10, int(base_symbol_size * scale_factor))
    try:
        font_scaled = ImageFont.truetype("arial.ttf", scaled_font_size)
    except IOError:
        font_scaled = ImageFont.load_default()
        
    for i, img in enumerate(unscaled_images):
        # Paste molecule/question mark
        sf_w, sf_h = int(img.width * scale_factor), int(img.height * scale_factor)
        # Ensure dimensions are positive
        sf_w = max(1, sf_w)
        sf_h = max(1, sf_h)
        
        img_to_paste = img.resize((sf_w, sf_h), Image.Resampling.LANCZOS)
        
        # Center vertically
        y_pos = (final_h - sf_h) // 2
        final_img.paste(img_to_paste, (int(x_offset), y_pos))
        x_offset += sf_w
        
        # Draw symbol
        if i < len(unscaled_images) - 1:
            symbol_text = symbols[symbol_idx]
            
            # Recalculate symbol positioning using the scaled font
            bbox = draw.textbbox((0, 0), symbol_text, font=font_scaled)
            text_w_s = bbox[2] - bbox[0]
            text_h_s = bbox[3] - bbox[1]
            
            # Center the symbol vertically
            y_pos_s = (final_h - text_h_s) // 2
            
            # Add a small buffer before drawing the symbol
            symbol_start_x = int(x_offset + 5 * scale_factor)
            draw.text((symbol_start_x, y_pos_s), symbol_text, font=font_scaled, fill=(0, 0, 0))
            
            # Advance offset by symbol width plus padding
            x_offset = symbol_start_x + text_w_s + int(5 * scale_factor)
            symbol_idx += 1
            
    return image_to_base64(final_img)

def get_smiles_from_name(name):
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

def generate_xml(questions, normalized_smiles_dict, lang):
    quiz = ET.Element('quiz')
    prompt = '<p>Dibuja la molécula faltante:</p>' if lang == 'es' else '<p>Draw the missing molecule:</p>'
    for i, q in enumerate(questions):
        # Usar SMILES normalizado si existe, si no usar el original
        # Dado que la interfaz de JSME actualiza q['missing_smiles'], lo usamos directamente.
        smiles_to_use = q['missing_smiles']
        
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        name_el = ET.SubElement(question, 'name')
        ET.SubElement(name_el, 'text').text = q['name']
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        
        # Escapar caracteres especiales para Moodle (Línea solicitada)
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
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8') # Cambio de tab a 2 espacios

# --- 5. BÚSQUEDA SIN SALTO (ANTES DEL UI) ---
def search_compound():
    name = st.session_state.search_input.strip()
    if not name:
        return
    with st.spinner("Buscando..."):
        smiles = get_smiles_from_name(name)
    if smiles:
        st.session_state.search_result = smiles
    else:
        st.error(texts["name_error"].format(name))

# --- 6. UI ---
st.set_page_config(page_title=texts["title"], layout="wide")

# Language switch
col_lang, _ = st.columns([1, 10])
with col_lang:
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

        # BÚSQUEDA SIN SALTO
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            st.text_input(
                texts["name_input_label"],
                key="search_input",
                label_visibility="collapsed",
                on_change=search_compound
            )
        with col_btn:
            if st.button(texts["search_button"], use_container_width=True):
                search_compound()

        # RESULTADO
        if st.session_state.get("search_result"):
            st.markdown(
                f"""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb; margin: 10px 0;">
                    <strong>{texts['smiles_found'].split(':')[0]}:</strong><br>
                    <code>{st.session_state.search_result}</code>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button(texts["add_to_reactants"], use_container_width=True, key="add_r"):
                    current = st.session_state.reactants_str
                    st.session_state.reactants_str = f"{current}, {st.session_state.search_result}".strip(", ")
                    st.session_state.search_result = None
                    st.rerun()
            with c2:
                if st.button(texts["add_to_products"], use_container_width=True, key="add_p"):
                    current = st.session_state.products_str
                    st.session_state.products_str = f"{current}, {st.session_state.search_result}".strip(", ")
                    st.session_state.search_result = None
                    st.rerun()

        st.markdown("---")

        reactants_str = st.text_area(
            texts["reactants_label"],
            value=st.session_state.reactants_str,
            height=80,
            placeholder="CC(=O)O, O"
        )
        products_str = st.text_area(
            texts["products_label"],
            value=st.session_state.products_str,
            height=80,
            placeholder="CO, CO2"
        )

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
        reaction_name = st.text_input(
            texts["reaction_name_label"],
            value=default_name,
            placeholder="Ej: Combustión de metano"
        )

        default_fb = '¡Muy bien!' if st.session_state.lang == 'es' else 'Well done!'
        correct_feedback = st.text_area(texts["correct_feedback_label"], value=default_fb)
        incorrect_feedback = st.text_area(texts["incorrect_feedback_label"], placeholder="Opcional")

        st.markdown("---")

        if st.button(texts["add_reaction_button"], type="primary", icon=":material/add_task:", use_container_width=True):
            if not reaction_name or not all_molecules or missing_index is None:
                st.error(texts["smiles_empty_error"])
            else:
                missing_smiles = all_molecules[missing_index]
                try:
                    if RDKIT_AVAILABLE:
                        for s in all_molecules:
                            # Simple validation (RDKit only needed if available)
                            if Chem.MolFromSmiles(s) is None and s not in ['[H+]', '[OH-]', 'H2O', 'O2', 'N2', 'CO2']:
                                st.error(texts["smiles_invalid_error"].format(s))
                                st.stop()
                    
                    # Call image generation (uses fixed NUMPY_AVAILABLE)
                    img = generate_reaction_image(reactants_list, products_list, missing_smiles)
                    
                    if not img:
                        st.error("Error al generar imagen o faltan dependencias.")
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
                    st.session_state.jsme_normalized = False  # Resetear normalización
                    st.session_state.normalized_smiles = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.markdown(texts["bulk_info"])
        uploaded = st.file_uploader(texts["upload_file_label"], type=['xlsx', 'csv'])
        if uploaded and st.button(texts["process_bulk_button"], type="primary", use_container_width=True):
            st.info("Carga masiva no implementada aún.")

# === OUTPUT COLUMN ===
with list_col:
    st.subheader(texts["added_questions_title"])

    if st.session_state.reaction_questions:
        # --- 1. BOTÓN: Normalizar con JSME ---
        if not st.session_state.jsme_normalized:
            if st.button(texts["normalize_button"], use_container_width=True, type="secondary"):
                st.session_state.show_jsme = True
                st.rerun()

        # --- EDITOR JSME ---
        if st.session_state.show_jsme:
            st.markdown("### Edita cada molécula")
            with st.form(key="jsme_form"):
                total = len(st.session_state.reaction_questions)
                for i, q in enumerate(st.session_state.reaction_questions):
                    st.markdown(f"**{i+1}. {q['name']}**")
                    # Nota: `jsme_editor` es un componente externo que no se puede incluir aquí.
                    # Asumiendo que `my_component` está disponible.
                    try:
                        jsme_editor(q['missing_smiles'], key=f"jsme_{i}")
                    except NameError:
                        st.warning("El componente `jsme_editor` no está disponible. Saltando edición.")

                submitted = st.form_submit_button("Aplicar Normalización", type="primary", use_container_width=True)
                if submitted:
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

        # --- 2. BOTÓN: Descargar XML ---
        if st.session_state.jsme_normalized:
            try:
                # El parámetro normalized_smiles_dict no es crítico ya que el SMILES está actualizado en questions
                xml_data = generate_xml(st.session_state.reaction_questions, st.session_state.normalized_smiles, st.session_state.lang)
                st.download_button(
                    label=texts["download_xml_button"],
                    data=xml_data,
                    file_name="reaction_questions.xml",
                    mime="application/xml",
                    use_container_width=True,
                    type="primary"
                )
            except Exception as e:
                st.error(texts["xml_error"].format(e))
        else:
            st.info(texts["normalize_first"])

        # --- Borrar todo ---
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
                    # Decode base64 for image display
                    try:
                        st.image(io.BytesIO(base64.b64decode(q['img_base64'])), width=250)
                    except:
                        st.warning("No se pudo mostrar la imagen (error de base64).")
                    st.caption(f"Faltante: `{q['missing_smiles']}`")
                with c2:
                    if st.button("🗑️", key=f"del_{i}"):
                        del st.session_state.reaction_questions[i]
                        st.rerun()
                st.markdown("---")
    else:
        st.info(texts["no_questions"])