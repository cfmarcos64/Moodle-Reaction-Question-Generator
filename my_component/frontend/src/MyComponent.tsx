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
  error: string | null;
}

class JSMEComponent extends StreamlitComponentBase<State> {
  private jsmeApplet: any = null;
  private isProcessing: boolean = false;
  private mountAttempted: boolean = false;

  constructor(props: any) {
    super(props);
    this.state = { 
      jsmeLoaded: false,
      processing: false,
      lastProcessedSmiles: "",
      error: null
    };
  }

  componentDidMount(): void {
    // CRÍTICO: Registrar el componente INMEDIATAMENTE
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(80);
    
    if (this.mountAttempted) {
      console.log("Component already mounted, skipping initialization");
      return;
    }
    
    this.mountAttempted = true;
    
    // Definir jsmeOnLoad si no existe
    if (!window.jsmeOnLoad) {
      window.jsmeOnLoad = () => {
        console.log("JSME library loaded via jsmeOnLoad");
      };
    }
    
    // Cargar JSME
    this.loadJSME();
  }

  componentWillUnmount(): void {
    this.mountAttempted = false;
  }

  loadJSME = (): void => {
    // Reutilizar instancia global si existe
    if (window.jsmeAppletInstance) {
      console.log("Using existing JSME instance");
      this.jsmeApplet = window.jsmeAppletInstance;
      this.setState({ jsmeLoaded: true });
      
      setTimeout(() => {
        const smiles = this.props.args?.smiles || '';
        if (smiles) {
          this.processSmiles(smiles);
        } else {
          Streamlit.setComponentValue("");
        }
      }, 100);
      return;
    }
    
    if (window.JSApplet) {
      console.log("JSApplet already available");
      this.initJSME();
      return;
    }

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
      this.setState({ error: "Error loading JSME library" });
      Streamlit.setComponentValue("");
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
      this.setState({ error: "Timeout loading JSME" });
      Streamlit.setComponentValue("");
      return;
    }

    setTimeout(() => this.waitForJSME(attempts + 1), 100);
  }

  initJSME = (): void => {
    try {
      console.log("Initializing JSME applet...");
      
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

      this.jsmeApplet = new window.JSApplet.JSME("jsme-hidden-container", "400px", "400px", {
        options: "oldlook,star"
      });

      window.jsmeAppletInstance = this.jsmeApplet;
      console.log("JSME applet created successfully");
      this.setState({ jsmeLoaded: true });
      
      setTimeout(() => {
        const smiles = this.props.args?.smiles || '';
        if (smiles && smiles.trim() !== '') {
          console.log("Processing initial SMILES:", smiles);
          this.processSmiles(smiles);
        } else {
          Streamlit.setComponentValue("");
        }
      }, 1000);

    } catch (error) {
      console.error("Error initializing JSME:", error);
      this.setState({ error: "Error initializing JSME" });
      Streamlit.setComponentValue("");
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
        this.isProcessing = false;
        this.setState({ processing: false });
        return;
      }

      this.jsmeApplet.clear();
      
      setTimeout(() => {
        try {
          this.jsmeApplet.readGenericMolecularInput(smiles);
          
          setTimeout(() => {
            try {
              const processedSmiles = this.jsmeApplet.smiles();
              console.log("Processed SMILES:", processedSmiles);
              
              if (processedSmiles && processedSmiles.trim() !== '') {
                Streamlit.setComponentValue(processedSmiles);
              } else {
                console.warn("Empty SMILES, using original");
                Streamlit.setComponentValue(smiles);
              }
            } catch (err) {
              console.error("Error getting SMILES:", err);
              Streamlit.setComponentValue(smiles);
            } finally {
              this.isProcessing = false;
              this.setState({ processing: false });
            }
          }, 500);
          
        } catch (err) {
          console.error("Error reading SMILES:", err);
          Streamlit.setComponentValue(smiles);
          this.isProcessing = false;
          this.setState({ processing: false });
        }
      }, 200);
      
    } catch (error) {
      console.error("Error processing SMILES:", error);
      Streamlit.setComponentValue(smiles);
      this.isProcessing = false;
      this.setState({ processing: false });
    }
  }

  public render(): ReactNode {
    const { jsmeLoaded, processing, error } = this.state;
    
    // Verificar si hay nuevos props y procesar
    const currentSmiles = this.props.args?.smiles || '';
    if (currentSmiles !== this.state.lastProcessedSmiles && 
        this.jsmeApplet && 
        this.state.jsmeLoaded && 
        !this.isProcessing) {
      // Programar procesamiento para el siguiente ciclo
      setTimeout(() => this.processSmiles(currentSmiles), 0);
    }

    if (error) {
      return (
        <div style={{
          padding: '8px 12px',
          backgroundColor: '#fee',
          borderRadius: '4px',
          textAlign: 'center',
          fontSize: '13px',
          color: '#c33'
        }}>
          ❌ {error}
        </div>
      );
    }

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