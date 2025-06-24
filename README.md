# Adaptive Transformer Trainer (Automated) 🚀

An intelligent PyTorch Transformer implementation from scratch with automated training, memory optimization, and dataset-aware preprocessing. This project converts notebook-based transformer training into production-ready Python scripts with smart GPU memory management and adaptive training strategies.

## 🌟 Key Features

- **🧠 Smart Memory Management**: Automatically detects GPU memory and adjusts parameters accordingly
- **📊 Dataset Intelligence**: Handles variable-length sequences through intelligent truncation
- **🔄 Automated Training Pipeline**: One-command training with progressive fallback strategies
- **💾 Low-Memory GPU Support**: Optimized for GPUs with 4GB+ memory (tested on GTX 1050)
- **🎯 Multiple Decoding Strategies**: Beam search, greedy decoding, and attention visualization
- **⚡ Progressive Fallback**: GPU → Reduced batch size → CPU training

## 🛠️ Project Structure

```
├── config.py                    # Configuration and hyperparameters
├── model.py                     # Transformer architecture implementation
├── train.py                     # Core training loop
├── dataset.py                   # Data loading with truncation support
├── runner.py                    # Automated training with memory optimization
├── translate.py                 # Translation and inference
├── attention_visual.py          # Attention mechanism visualization
├── Beam_Search.py               # Beam search implementation
├── inference.py                 # Model inference utilities
├── Local_Train.py               # Local training configuration
├── memory_optimized_train.py    # Memory-optimized training scripts
├── cpu_fallback_train.py        # CPU fallback training
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### Installation

```
git clone https://github.com/Ashwin-Baduni/adaptive-transformer-trainer-automated.git
cd adaptive-transformer-trainer-automated
pip install -r requirements.txt
```

### Training

- **Automatic training with memory optimization**
  ```
  python runner.py --seq-len 200 --batch-size 1 --epochs 20
  ```
- **Quick test (1 epoch)**
  ```
  python runner.py --seq-len 100 --batch-size 1 --epochs 1
  ```
- **CPU fallback if GPU memory insufficient**
  ```
  python runner.py --cpu-only --epochs 5
  ```

### Translation

```
python translate.py
# or
python inference.py
```

## 💡 What Makes This Special

### 🧠 Intelligent Memory Management

- **Auto-Detection**: Automatically detects GPU memory (e.g., 3.94GB GTX 1050)
- **Dynamic Adjustment**: Adjusts batch size, sequence length, and model dimensions
- **Progressive Fallback**: GPU → Smaller parameters → CPU training

### 📊 Dataset-Aware Processing

- **Smart Truncation**: Handles sentences up to 309 tokens through intelligent truncation
- **Quality Preservation**: Maintains translation quality while fitting memory constraints
- **Flexible Data Sources**: Supports any language pair through Hugging Face datasets

### 🔄 Automated Training Pipeline

- **One-Command Training**: Complete training setup with optimal configurations
- **Checkpoint Management**: Automatic saving and resuming from checkpoints
- **Monitoring Integration**: TensorBoard logging for training visualization

## 📈 Training Results

Successfully tested on GTX 1050 (4GB VRAM):

- **Dataset**: OPUS Books (English → Italian, 32,332 pairs)
- **Training Speed**: ~14 iterations/second
- **Memory Efficiency**: Optimized for 4GB GPU memory
- **Sequence Handling**: 100-200 tokens (configurable with truncation)
- **Training Time**: ~34 minutes per epoch

## 🎯 Use Cases

- **🔬 Research**: Experiment with transformer architectures on limited hardware
- **📚 Education**: Learn transformer implementation with practical constraints
- **🚀 Production**: Deploy on resource-constrained environments
- **📊 Custom Data**: Train on your own translation datasets

## 🔧 Configuration

### Basic Configuration

Modify `config.py` for different setups:

```
{
    'datasource': 'opus_books',     # Dataset source
    'lang_src': 'en',              # Source language
    'lang_tgt': 'it',              # Target language
    'seq_len': 200,                # Sequence length
    'batch_size': 1,               # Batch size
    'num_epochs': 20,              # Training epochs
    'd_model': 256,                # Model dimension
}
```

### Memory Optimization

The system automatically optimizes based on available GPU memory.

## 🙏 Acknowledgments

- **Based on the "Attention Is All You Need" paper**
- **Inspired by Hugging Face Transformers**
- **Optimized for educational and research purposes**
- **Built with PyTorch and modern ML best practices**

## 📞 Contact

**Ashwin Baduni**  
Email: baduniashwin@gmail.com
