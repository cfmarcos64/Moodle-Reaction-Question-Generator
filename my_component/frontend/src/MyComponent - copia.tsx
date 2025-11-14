import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

declare global {
  interface Window {
    JSApplet: any;
    jsmeOnLoad?: () => void;
    jsmeAppletInstance?: any;
  }
}

interface State {
  jsmeLoaded: boolean;
  processing: boolean;
  lastProcessedSmiles: string;
}

class JSMEComponent extends StreamlitComponentBase<State> {
  private jsmeApplet: any = null;
  private isProcessing: boolean = false;
  private hasInitialized: boolean = false;

  constructor(props: any) {
    super(props);
    this.state = { 
      jsmeLoaded: false,
      processing: false,
      lastProcessedSmiles: ""
    };
  }

  componentDidMount(): void {
    // Evitar inicialización múltiple
    if (this.hasInitialized) {
      console.log("Already initialized, skipping...");
      return;
    }
    
    this.hasInitialized = true;
    
    // Definir la función global jsmeOnLoad que JSME espera
    if (!window.jsmeOnLoad) {
      window.jsmeOnLoad = () => {
        console.log("JSME library loaded via jsmeOnLoad callback");
      };
    }
    
    // Notificar a Streamlit que el componente está listo PRIMERO
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(80);
    
    // Luego cargar JSME
    this.loadJSME();
  }

  componentWillUnmount(): void {
    this.hasInitialized = false;
  }

  render(): void {
    const currentSmiles = this.props.args?.smiles || '';
    
    // Solo procesar si cambió y JSME está listo
    if (currentSmiles !== this.state.lastProcessedSmiles && 
        this.jsmeApplet && 
        this.state.jsmeLoaded && 
        !this.isProcessing) {
      this.processSmiles(currentSmiles);
    }
  }

  loadJSME = (): void => {
    // Si ya existe una instancia global, reutilizarla
    if (window.jsmeAppletInstance) {
      console.log("Using existing JSME instance");
      this.jsmeApplet = window.jsmeAppletInstance;
      this.setState({ jsmeLoaded: true });
      
      setTimeout(() => {
        const smiles = this.props.args?.smiles || '';
        if (smiles) {
          this.processSmiles(smiles);
        }
      }, 100);
      return;
    }
    
    // Verificar si JSME ya está cargado
    if (window.JSApplet) {
      console.log("JSME already loaded");
      this.initJSME();
      return;
    }

    // Verificar si el script ya existe
    const existingScript = document.querySelector('script[src*="jsme.nocache.js"]');
    if (existingScript) {
      console.log("JSME script already in DOM, waiting...");
      this.waitForJSME();
      return;
    }

    console.log("Loading JSME script...");
    const script = document.createElement('script');
    script.src = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js';
    script.async = true;
    script.onload = () => {
      console.log("JSME script loaded");
      this.waitForJSME();
    };
    script.onerror = (error) => {
      console.error("Error loading JSME:", error);
    };
    
    document.head.appendChild(script);
  }

  waitForJSME = (attempts: number = 0): void => {
    if (window.JSApplet) {
      console.log("JSApplet available, initializing...");
      this.initJSME();
      return;
    }

    if (attempts > 100) {
      console.error("Timeout waiting for JSME");
      return;
    }

    setTimeout(() => this.waitForJSME(attempts + 1), 100);
  }

  initJSME = (): void => {
    try {
      console.log("Initializing JSME applet...");
      
      // Crear contenedor oculto si no existe
      let hiddenDiv = document.getElementById('jsme-hidden-container');
      if (!hiddenDiv) {
        hiddenDiv = document.createElement('div');
        hiddenDiv.id = 'jsme-hidden-container';
        hiddenDiv.style.display = 'none';
        hiddenDiv.style.position = 'absolute';
        hiddenDiv.style.left = '-9999px';
        hiddenDiv.style.width = '400px';
        hiddenDiv.style.height = '400px';
        document.body.appendChild(hiddenDiv);
      }

      // Crear el applet JSME con opciones más permisivas
      this.jsmeApplet = new window.JSApplet.JSME("jsme-hidden-container", "400px", "400px", {
        options: "oldlook,star"
      });

      // Guardar instancia global
      window.jsmeAppletInstance = this.jsmeApplet;

      console.log("JSME applet created successfully");
      this.setState({ jsmeLoaded: true });
      
      // Esperar más tiempo para que JSME se inicialice completamente
      setTimeout(() => {
        const smiles = this.props.args?.smiles || '';
        if (smiles && smiles.trim() !== '') {
          console.log("Processing initial SMILES:", smiles);
          this.processSmiles(smiles);
        } else {
          Streamlit.setComponentValue("");
        }
      }, 1000); // Aumentado a 1 segundo

    } catch (error) {
      console.error("Error initializing JSME:", error);
    }
  }

  processSmiles = (smiles: string): void => {
    if (this.isProcessing) {
      console.log("Already processing, skipping...");
      return;
    }

    this.isProcessing = true;
    this.setState({ processing: true, lastProcessedSmiles: smiles });

    try {
      console.log("Processing SMILES:", smiles);
      
      if (!smiles || smiles.trim() === '') {
        Streamlit.setComponentValue("");
        return;
      }

      // Limpiar el applet primero
      this.jsmeApplet.clear();
      
      // Esperar un momento antes de cargar
      setTimeout(() => {
        try {
          // Cargar el SMILES en JSME
          this.jsmeApplet.readGenericMolecularInput(smiles);
          
          // Esperar a que JSME procese
          setTimeout(() => {
            try {
              // Obtener el SMILES procesado
              const processedSmiles = this.jsmeApplet.smiles();
              console.log("Raw processed SMILES:", processedSmiles);
              
              // Verificar si realmente obtuvimos algo
              if (processedSmiles && processedSmiles.trim() !== '') {
                console.log("Valid processed SMILES:", processedSmiles);
                Streamlit.setComponentValue(processedSmiles);
              } else {
                console.warn("JSME returned empty SMILES, using original");
                Streamlit.setComponentValue(smiles);
              }
            } catch (err) {
              console.error("Error getting SMILES from JSME:", err);
              Streamlit.setComponentValue(smiles);
            } finally {
              this.isProcessing = false;
              this.setState({ processing: false });
            }
          }, 300); // Dar tiempo a JSME para procesar
          
        } catch (err) {
          console.error("Error reading SMILES into JSME:", err);
          Streamlit.setComponentValue(smiles);
          this.isProcessing = false;
          this.setState({ processing: false });
        }
      }, 100);
      
    } catch (error) {
      console.error("Error processing SMILES:", error);
      Streamlit.setComponentValue(smiles);
      this.isProcessing = false;
      this.setState({ processing: false });
    }
  }

  public render = (): ReactNode => {
    const { jsmeLoaded, processing } = this.state;

    let statusText = '⏳ Cargando JSME...';
    let bgColor = '#f0f0f0';
    
    if (jsmeLoaded && !processing) {
      statusText = '✓ JSME listo';
      bgColor = '#e8f5e9';
    } else if (processing) {
      statusText = '⚙️ Procesando...';
      bgColor = '#fff3e0';
    }

    return (
      <div style={{
        padding: '8px 12px',
        backgroundColor: bgColor,
        borderRadius: '4px',
        textAlign: 'center',
        fontSize: '13px',
        color: '#666',
        transition: 'background-color 0.3s'
      }}>
        {statusText}
      </div>
    );
  }
}

export default withStreamlitConnection(JSMEComponent)