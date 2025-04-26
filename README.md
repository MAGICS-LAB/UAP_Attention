# Universal Approximation with Softmax Attention

----------

This repo is the official implementation of experiments in our paper:  
**[Universal Approximation with Softmax Attention
](https://arxiv.org/abs/2504.15956)** by  
Jerry Yao-Chieh Hu*, Hude Liu*, Hong-Yu Chen*, Weimin Wu, Han Liu



### **Instruction**
The code was tested on `Python 3.11`. To install, run `pip install -r requirements.txt`.
#### Approximation Rate Validation for Theorem 3.1/Theorem 3.2
Run `truncated.py`. 

#### Attention Heatmap for different $|b-a|$
Run `attn_map.py`.

#### Sequence-to-Sequence Approximation
First generate data by running `seq2seq_data.py`, then run `seq2seq.py`.

----------


### **Citation**

If you have any question regarding our paper or codes, please feel free to start an issue or email Hong-Yu Chen (charlie.chen@u.northwestern.edu).
If you find our work useful, please kindly cite our paper:
```
@article{hu2025universal,
  title={Universal Approximation with Softmax Attention},
  author={Hu, Jerry Yao-Chieh and Liu, Hude and Chen, Hong-Yu and Wu, Weimin and Liu, Han},
  journal={arXiv preprint arXiv:2504.15956},
  year={2025}
}
```

