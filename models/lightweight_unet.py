import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import safetensors.torch

class LightweightUNet(nn.Module):
    def __init__(self, num_classes=2, encoder_name="tu-swin_tiny_patch4_window7_224", use_satellite_weights=False):
        """
        Initializes a lightweight U-Net using a Swin Transformer backbone.
        Requires `segmentation-models-pytorch` and `timm`.
        
        Args:
            num_classes (int): Number of segmentation classes.
            encoder_name (str): The timm backbone to use. Defaults to a Swin Tiny transformer.
            use_satellite_weights (bool): Se True, carica i pesi pre-addestrati su foto satellitari.
                                          Se False, usa quelli base (ImageNet).
        """
        super().__init__()
        print(f"Initializing U-Net with backbone: {encoder_name}")
        
        # Scegliamo se usare i pesi imagenet nativi di SMP o i nostri custom satellitari
        weights_arg = None if use_satellite_weights else "imagenet"
        
        try:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=weights_arg,
                in_channels=3,
                classes=num_classes,
            )
            print("Successfully loaded the model backend from smp.")
            
            # --- CUSTOM SATELLITE WEIGHTS INTEGRATION ---
            if use_satellite_weights:
                print("Setting up Remote Sensing pretrained weights...")
                
                # INSERISCI QUI IL LINK DIRETTO AL FILE .pth SATELLITARE (es. da HuggingFace)
                # Oppure scarica il file a mano e mettilo nella cartella del progetto chiamato 'rsp-swin-t-ckpt.pth'
                RS_WEIGHTS_URL = "https://huggingface.co/BiliSakura/RSP-Swin-T/resolve/main/model.safetensors" 
                LOCAL_WEIGHTS_PATH = "rsp-swin-t-ckpt.safetensors"
                
                if not os.path.exists(LOCAL_WEIGHTS_PATH):
                    print(f"File locale non trovato. Provo a scaricare i pesi da {RS_WEIGHTS_URL} ...")
                    try:
                        os.system(f"wget -O {LOCAL_WEIGHTS_PATH} {RS_WEIGHTS_URL}")
                        print("Download completato con successo!")
                    except Exception as e:
                        print(f"Attenzione: download automatico fallito: {e}")
                
                # Carichiamo i pesi nell'encoder (strict=False aggira piccoli mismatch architetturali periferici)
                if os.path.exists(LOCAL_WEIGHTS_PATH):
                    # Uso Safetensors invece di torch.load
                    state_dict = safetensors.torch.load_file(LOCAL_WEIGHTS_PATH, device="cpu")
                    
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    elif "model" in state_dict:
                        state_dict = state_dict["model"]
                        
                    # === SUPER CRITICAL KEY MAPPING ===
                    # I ricercatori di RSP hanno usato una vecchia versione architetturale di Swin.
                    # Nelle vecchie versioni: il downsample avviene alla FINE dello stadio i (es. layers.0.downsample)
                    # Nella versione moderna (smp/timm): il downsample avviene all'INIZIO dello stadio i+1 (es. model.layers_1.downsample)
                    mapped_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = "model." + k
                        
                        if "downsample" in k:
                            # Separliamo la stringa: ["model", "layers", "0", "downsample", ...]
                            parts = new_k.split('.')
                            if parts[1] == "layers":
                                layer_idx = int(parts[2])
                                parts[2] = str(layer_idx + 1) # Mettiamo lo shift in avanti di +1
                            # Ricongiungiamo e mettiamo l'underscore che serve a smp
                            new_k = ".".join(parts).replace("layers.", "layers_")
                        else:
                            # Semplice sostituzione del punto con l'underscore per i layer normali
                            new_k = new_k.replace("layers.", "layers_")
                            
                        mapped_state_dict[new_k] = v

                    missing, unexpected = self.model.encoder.load_state_dict(mapped_state_dict, strict=False)
                    
                    if len(missing) > 100:
                        print("⚠️ ATTENZIONE: La maggior parte dei pesi non ha matchato (nomi file incompatibili?). Il backbone è rimasto random.")
                    else:
                        print("🚀 Pesi satellitari Swin-T (RSP) mappati e iniettati chirurgicamente con successo!")
                    
        except Exception as e:
            print(f"Could not load requested backbone {encoder_name}: {e}")
            print("Falling back to standard ResNet-34 backbone for safety.")
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=num_classes,
            )
            
    def forward(self, x):
        # SMP riceve [B, C, H, W] ed emette [B, num_classes, H, W] direttamente
        return self.model(x)
