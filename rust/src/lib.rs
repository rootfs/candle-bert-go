use std::ffi::{c_char, CStr, CString};
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Structure to hold BERT model, tokenizer, and classification head
struct BertClassifier {
    model: BertModel,
    tokenizer: Tokenizer,
    classification_head: Linear,
    num_classes: usize,
    device: Device,
}

lazy_static::lazy_static! {
    static ref BERT_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
}

impl BertClassifier {
    fn new(model_id: &str, num_classes: usize, use_cpu: bool) -> Result<Self> {
        if num_classes < 2 {
            return Err(E::msg(format!("Number of classes must be at least 2, got {}", num_classes)));
        }

        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load model and tokenizer from Hugging Face Hub
        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        );

        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use approximate GELU for better performance
        config.hidden_act = HiddenAct::GeluApproximate;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;

        // Create a classification head with non-zero initial parameters for better starting point
        let hidden_size = config.hidden_size;
                
        // Transpose the weight matrix to match the expected dimensions
        let w = Tensor::randn(0.0, 0.02, (hidden_size, num_classes), &device)?;
        let b = Tensor::zeros((num_classes,), DType::F32, &device)?;
        
        let classification_head = candle_nn::Linear::new(w, Some(b));

        Ok(Self {
            model,
            tokenizer,
            classification_head,
            num_classes,
            device,
        })
    }

    fn classify_text(&self, text: &str) -> Result<(usize, f32)> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(E::msg)?;
        
        let token_ids = encoding.get_ids().to_vec();
        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        
        // Run the text through BERT
        let embeddings = self.model.forward(&token_ids_tensor, &token_type_ids, None)?;
        
        // Extract the CLS token (first token) embedding for classification
        let cls_embedding = embeddings.narrow(1, 0, 1)?.squeeze(1)?;
        
        // Get the dimensions and convert to the right type
        let hidden_size = cls_embedding.dim(1)?;
        let cls_embedding = cls_embedding.to_dtype(DType::F32)?;
        
        // Access the weights and reshape them for correct matrix multiplication
        let weights = self.classification_head.weight().to_dtype(DType::F32)?;
        let bias = self.classification_head.bias().unwrap().to_dtype(DType::F32)?;
        
        // Make sure the weight dimensions are correct for matmul
        // For matmul of shapes [1, hidden_size] and [hidden_size, num_classes]
        // the weight must be [hidden_size, num_classes]
        let w_reshaped = weights.reshape((hidden_size, self.num_classes))?;
        
        // Perform matrix multiplication: [1, hidden_size] x [hidden_size, num_classes] -> [1, num_classes]
        let logits = cls_embedding.matmul(&w_reshaped)?;
        
        // Add bias
        let logits = logits.broadcast_add(&bias)?;
        
        // If logits has shape [1, num_classes], squeeze it to get [num_classes]
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits
        };
        
        // Get the class with highest probability (argmax)
        let class_idx = logits.argmax(0)?
            .to_dtype(DType::U32)?
            .to_vec0::<u32>()?;
        
        // Ensure we don't return a class index outside our expected range
        if class_idx as usize >= self.num_classes {
            return Err(E::msg(format!("Invalid class index: {} (num_classes: {})", 
                                      class_idx, self.num_classes)));
        }
            
        // Compute softmax probabilities
        let logits_vec = logits.to_vec1::<f32>()?;
        let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits_vec.iter().map(|&x| (x - max_logit).exp()).sum();
        
        // Calculate confidence
        let confidence = (logits_vec[class_idx as usize] - max_logit).exp() / exp_sum;
        
        Ok((class_idx as usize, confidence))
    }
}

// Initialize the BERT model (called from Go)
#[no_mangle]
pub extern "C" fn init_bert(model_id: *const c_char, use_cpu_or_num_classes: i32, use_cpu_maybe: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Check if use_cpu_or_num_classes is a boolean or number of classes
    // If it's 0 or 1, treat it as a boolean (use_cpu)
    // If it's >= 2, treat it as number of classes
    let (num_classes, use_cpu) = if use_cpu_or_num_classes <= 1 {
        // Original behavior - 2 classes with use_cpu parameter
        (2, use_cpu_or_num_classes == 1)
    } else {
        // New behavior - multi-class with separate use_cpu parameter
        (use_cpu_or_num_classes as usize, use_cpu_maybe)
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2");
        return false;
    }

    match BertClassifier::new(model_id, num_classes, use_cpu) {
        Ok(classifier) => {
            let mut bert_opt = BERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT: {}", e);
            false
        }
    }
}

// New structure to hold classification result
#[repr(C)]
pub struct ClassificationResult {
    pub class: i32,
    pub confidence: f32,
}

// Classify text using BERT (called from Go) - returns only class index
#[no_mangle]
pub extern "C" fn classify_text(text: *const c_char) -> i32 {
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    let bert_opt = BERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, _confidence)) => class_idx as i32,
            Err(e) => {
                eprintln!("Error classifying text: {}", e);
                -1
            }
        },
        None => {
            eprintln!("BERT model not initialized");
            -1
        }
    }
}

// Classify text using BERT (called from Go) - returns both class and confidence
#[no_mangle]
pub extern "C" fn classify_text_with_confidence(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult { class: -1, confidence: 0.0 };
    
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    let bert_opt = BERT_CLASSIFIER.lock().unwrap();
    match &*bert_opt {
        Some(classifier) => match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                class: class_idx as i32,
                confidence: confidence,
            },
            Err(e) => {
                eprintln!("Error classifying text: {}", e);
                default_result
            }
        },
        None => {
            eprintln!("BERT model not initialized");
            default_result
        }
    }
}

// Free a C string allocated by Rust (called from Go)
#[no_mangle]
pub extern "C" fn free_cstring(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
} 