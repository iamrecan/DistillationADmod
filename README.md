<p align="center">
  <h1><center> &#127981;&#9879; Distillation Industrial Anomaly Detection and Understanding Analysis Error &#9879;&#127981; </center></h1>
</p>

# Description
This project aims to regroup the state-of-the-art approaches that use knowledge distillation for unsupervised anomaly detection. The code is designed to be understandable and simple to allow custom modifications.

## 🚀 New Features

### 🎯 Integrated Anomaly Detection System
- **Full Pipeline**: Anomaly Detection → SAM2 Segmentation → LLM Analysis
- **Multiple Models**: Supports SN, DBFAD, EAD, RD, ST model types
- **Adaptive Thresholding**: Smart threshold calibration based on normal images
- **Interactive Workflow**: User confirmation at each step
- **Comprehensive Reporting**: Detailed JSON reports with analysis results

### 🤖 SAM2 + Google AI Integration
- **SAM2 Segmentation**: Automated segmentation based on anomaly maps
- **Google AI Analysis**: Advanced LLM-based defect analysis
- **Multi-modal Analysis**: Combined visual and text-based insights

### 📊 Analysis Configuration Management
- **Centralized Config**: All analysis paths and settings in `analysis_config.py`
- **Organized Reports**: Separate directories for different report types
- **Auto-cleanup**: Automatic old file management
- **Migration Support**: Moves existing reports to new structure

## Getting Started

You will need [Python 3.10+](https://www.python.org/downloads) and the packages specified in _requirements.txt_.

Install packages with:

```bash
pip install -r requirements.txt
```

## 🎯 Integrated Analysis Usage

### Quick Start with Integrated System
```bash
# Run full pipeline with interactive mode
python integrated_anomaly_system.py

# Run with specific parameters
python integrated_anomaly_system.py dataset/wood/test/hole/002.png sn wood
```

### SAM2 + Google AI Analysis
```bash
# Basic analysis
python sam2_google_ai_pipeline.py image.jpg

# Wood inspection with interactive prompts
python sam2_google_ai_pipeline.py image.jpg wood_inspection --interactive
```

### Report Analysis
```bash
# Interpret analysis reports
python report_interpreter.py report_file.json

# Debug analysis differences
python debug_analysis.py
```

## 📁 Project Structure

### Core Files
- `integrated_anomaly_system.py` - Main integrated pipeline
- `sam2_google_ai_pipeline.py` - SAM2 + Google AI pipeline
- `analysis_config.py` - Centralized analysis configuration
- `llm_image_analysis.py` - Google AI image analysis
- `report_interpreter.py` - Report analysis and interpretation

### Configuration
- `analysis_config.py` - Analysis paths and settings
- `configs/` - Model-specific configurations
- `config.yaml` - Main configuration file

### Reports and Results
- `reports/` - Organized analysis reports
  - `integrated/` - Full pipeline reports
  - `sam2/` - SAM2 segmentation reports
  - `llm/` - LLM analysis reports
  - `debug/` - Debug analysis reports
- `results/` - Visualization outputs
- `temp_analysis/` - Temporary analysis files

## Base usage 

### Configuration
To use the project, you must configure the config.yaml file 
This file allows configuring the main elements of the project.

- `data_path` (STR): The path to the dataset
- `distillType` (STR): The type of distillation : st for STPM, rd for reverse distillation, ead for EfficientAD, dbfad for distillation-based fabric anomaly detection, mixed for mixedTeacher, rnst/rnrd for remembering normality (forward/backward), sn for singlenet
- `backbone` (STR): The name of the model backbone (any CNN for st, only resnets and wide resnets for rd, small or medium for ead)
- `out_indice` (LIST OF INT): The index of the layer used for distillation (only for st)
- `obj` (STR): The object category
- `phase` (STR): Either train or test
- `save_path` (STR): The path to save the model weights
- `training_data`(YAML LIST) : To configure hyperparameters (epochs, batch_size, img_size, crop_size, norm and other parameters)

An example of config for each distillType is accessible in `configs/`

### Training and testing
Once configured, just do the following command to train or test (depending of configuration file)
```bash
python3 train.py
```

You can also visualize the feature map of a given layer, you may change the selected layer within the python file
```bash
python3 visualization.py
```

## 🧠 Advanced Features

### Adaptive Threshold Calibration
The system automatically calibrates thresholds based on normal images:
- **Percentile-based**: Uses 95th percentile of normal image scores
- **Statistical**: Mean + 2.5 × standard deviation
- **IQR-based**: Q3 + 1.5 × Interquartile Range
- **Combined approach**: Intelligently combines multiple methods

### Multi-Model Support
- **SingleNet (SN)**: Single-layer distillation with Fourier convolutions
- **DBFAD**: Distillation-based fabric anomaly detection
- **EfficientAD**: Millisecond-level accurate anomaly detection
- **ReverseDistillation (RD)**: One-class embedding approach
- **StudentTeacher (ST)**: Feature pyramid matching (STPM)

### Intelligent Analysis Pipeline
1. **Anomaly Detection**: Uses trained models with adaptive thresholding
2. **SAM2 Segmentation**: Focuses on detected anomaly regions
3. **LLM Analysis**: Provides detailed defect classification and recommendations
4. **User Interaction**: Confirmation points for human oversight

### Wood Quality Assessment
Specialized analysis for wood inspection:
- **Defect Detection**: Knots, cracks, discoloration, rot
- **Quality Grading**: A/B/C/D classification system
- **Usage Recommendations**: Structural vs. non-structural applications
- **Processing Suggestions**: Treatment and handling recommendations

## 📊 Report System

### Report Types
- **Integrated Reports**: Complete pipeline results
- **SAM2 Reports**: Segmentation analysis
- **LLM Reports**: AI analysis results
- **Debug Reports**: Troubleshooting information

### Report Features
- **JSON Format**: Machine-readable structured data
- **Human-Readable**: Interpreted summaries
- **Visualization**: Anomaly maps and segmentation results
- **Recommendations**: Actionable insights

### Report Management
- **Auto-organization**: Reports sorted by type and date
- **Cleanup**: Automatic old file removal
- **Migration**: Existing reports moved to new structure
- **Search**: Easy report discovery and analysis


## License

This project is licensed under the MIT License.