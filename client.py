from tensorflow.contrib import predictor
import tokenization_google as tokenization 
import pdb

class AlbertEncoder:
    def __init__(self,):
        self.max_seq_length = 128
        self.mode_dir = './model/1576043718'
        self.vocab_file = './model/vocab.txt'
        self.model = predictor.from_saved_model(self.mode_dir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        pass

    def process(self, text):
        #预处理
        text = tokenization.convert_to_unicode(text)
        text = self.tokenizer.tokenize(text)
        if len(text) > self.max_seq_length:
            text = text[0:self.max_seq_length-2]

        #输入id化
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in text:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = 0.0#模拟label

        output = self.model({'a_input_ids':[input_ids],
        'a_input_mask':[input_mask],
        'a_segment_ids':[segment_ids],
        'b_input_ids':[input_ids],
        'b_input_mask':[input_mask],
        'b_segment_ids':[segment_ids],
        'label_ids':[label_id]})['a_output_layer']

        return output
        pass

if __name__ == '__main__':
    encoder = AlbertEncoder()
    import time
    NUMBER = 100
    start_time = time.time()
    for i in range(NUMBER):
        encoder.process('测试啊哈哈')
    end_time = time.time()
    print("avg cost:%f"%((end_time - start_time)/NUMBER))
    pass

