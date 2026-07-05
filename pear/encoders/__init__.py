from .xlmr import XLMREncoder
from .xlmr_xl import XLMRXLEncoder

str2encoder = {
    "XLM-RoBERTa": XLMREncoder,
    "XLM-RoBERTa-XL": XLMRXLEncoder,
    "InfoXLM": XLMREncoder,
}
