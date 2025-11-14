import os
import streamlit.components.v1 as components

# Cambiar a True para producción (Streamlit Cloud)
# Cambiar a False para desarrollo local
_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "jsme_editor",
        url="http://localhost:3001",  # Para desarrollo local con npm run start
    )
else:
    # Para producción (Streamlit Cloud)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("jsme_editor", path=build_dir)


def jsme_editor(smiles, key=None):
    """
    Crea una instancia del editor molecular JSME para procesar SMILES.
    
    El componente JSME funciona en segundo plano (oculto) para normalizar
    y validar códigos SMILES, asegurando compatibilidad con Moodle.
    
    Parameters
    ----------
    smiles : str
        Código SMILES de la molécula a procesar
    key : str or None
        Clave única para identificar el componente. Importante cuando se usan
        múltiples instancias del componente en la misma aplicación.
    
    Returns
    -------
    str
        El código SMILES procesado y normalizado por JSME.
        Retorna cadena vacía si el procesamiento falla.
    
    Example
    -------
    >>> from my_component import jsme_editor
    >>> 
    >>> # Procesar un SMILES
    >>> original_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    >>> processed_smiles = jsme_editor(original_smiles, key="molecule_1")
    >>> print(processed_smiles)
    """
    component_value = _component_func(smiles=smiles, key=key, default="")
    
    return component_value