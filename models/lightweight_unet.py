import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LightweightUNet(nn.Module):
    def __init__(self, num_classes=2, encoder_name="tu-swin_tiny_patch4_window7_224", use_satellite_weights=True):
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
                RS_WEIGHTS_URL = "INSERISCI_QUI_IL_LINK_DIRETTO_HUGGINGFACE.pth" 
                LOCAL_WEIGHTS_PATH = "rsp-swin-t-ckpt.pth"
                
                if not os.path.exists(LOCAL_WEIGHTS_PATH):
                    print(f"File locale non trovato. Provo a scaricare i pesi da {RS_WEIGHTS_URL} ...")
                    try:
                        torch.hub.download_url_to_file(RS_WEIGHTS_URL, LOCAL_WEIGHTS_PATH)
                        print("Download completato con successo!")
                    except Exception as e:
                        print(f"Attenzione: download automatico fallito (link mancante o errato): {e}")
                        print("Per favore, scarica il file .pth satellitare a mano e salvalo come 'rsp-swin-t-ckpt.pth' nella stessa cartella.")
                
                # Carichiamo i pesi nell'encoder (strict=False aggira piccoli mismatch architetturali periferici)
                if os.path.exists(LOCAL_WEIGHTS_PATH):
                    # PyTorch 2.6+ blocca custom objects per sicurezza. Siccome RSP usa yacs.config.CfgNode, forziamo weights_only=False
                    state_dict = torch.load(LOCAL_WEIGHTS_PATH, map_location="cpu", weights_only=False)
                    
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    elif "model" in state_dict:
                        state_dict = state_dict["model"]
                        
                    # === CRITICAL KEY MAPPING ===
                    # RSP weights use raw Swin names (e.g. layers.0.blocks.0.norm1.weight)
                    # SMP Unet encoder expects "model." prefix and underscores for layers (e.g. model.layers_0.blocks...)
                    mapped_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = "model." + k
                        new_k = new_k.replace("layers.0", "layers_0")
                        new_k = new_k.replace("layers.1", "layers_1")
                        new_k = new_k.replace("layers.2", "layers_2")
                        new_k = new_k.replace("layers.3", "layers_3")
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
