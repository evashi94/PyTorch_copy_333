from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division
import numpy as np
from random import randint, choice
import torch

from core.env import Env

class AssociativeRecallEnv(Env):
    def __init__(self, args, env_ind=0):
        super(AssociativeRecallEnv, self).__init__(args, env_ind)

        # state space setup
        self.batch_size = args.batch_size
        self.len_word = args.len_word
        self.min_num_words = args.min_num_words
        self.max_num_words = args.max_num_words
        self.min_num_items = args.min_num_items
        self.max_num_items = args.max_num_items
        self.save_bit_error = args.save_bit_error
        self.save_avg_bit_error = args.save_avg_bit_error
        self.refs = args.refs
        self.logger.warning("Word     {length}:   {%s}", self.len_word)
        self.logger.warning("Words #  {min, max}: {%s, %s}", self.min_num_words, self.max_num_words)
        self.logger.warning("Item #  {min, max}: {%s, %s}", self.min_num_items, self.max_num_items)

    def _preprocessState(self, state):
        # NOTE: state input in size: batch_size x num_words  x len_word
        # NOTE: we return as:        num_words  x batch_size x len_word
        # NOTE: to ease feeding in one row from all batches per forward pass
        for i in range(len(state)):
            state[i] = np.transpose(state[i], (1, 0, 2))
        return state

    @property
    def state_shape(self):
        # NOTE: we use this as the input_dim to be consistent with the sl & rl tasks
        return self.len_word + 2

    @property
    def action_dim(self):
        # NOTE: we use this as the output_dim to be consistent with the sl & rl tasks
        return self.len_word

    def render(self):
        pass

    def _readable(self, datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'    

    def _seperate(self, datum): #not for reading
        return ['-' if x == 0 else '%d' % x for x in datum]

    def visual(self, input_ts, target_ts, mask_ts, output_ts=None):
        """
        input_ts:  [(num_wordsx2+2) x batch_size x (len_wordx2+2)]
        target_ts: [(num_wordsx2+2) x batch_size x (len_word)]
        mask_ts:   [(num_wordsx2+2) x batch_size x (len_word)]
        output_ts: [(num_wordsx2+2) x batch_size x (len_word)]
        """
        output_ts = torch.round(output_ts * mask_ts) if output_ts is not None else None
        input_strings  = [self._readable(input_ts[:, 0, i])  for i in range(input_ts.size(2))]
        target_strings = [self._readable(target_ts[:, 0, i]) for i in range(target_ts.size(2))]
        mask_strings   = [self._readable(mask_ts[:, 0, 0])]
        output_strings = [self._readable(output_ts[:, 0, i]) for i in range(output_ts.size(2))] if output_ts is not None else None

        # strings = [input_strings, target_strings, mask_strings, output_strings]
        # self.logger.warning(input_strings)
        # self.logger.warning(target_strings)
        # self.logger.warning(mask_strings)
        # self.logger.warning(output_strings)
        input_strings  = 'Input:\n'  + '\n'.join(input_strings)
        target_strings = 'Target:\n' + '\n'.join(target_strings)
        mask_strings   = 'Mask:\n'   + '\n'.join(mask_strings)
        output_strings = 'Output:\n' + '\n'.join(output_strings) if output_ts is not None else None
        """
        print(input_strings)
        print(target_strings)
        print(mask_strings)
        print(output_strings) if output_ts is not None else None
        """

        #added on 28 Oct
        #print("------------bit error called here")
        target_list = [self._seperate(target_ts[:, 0, i]) for i in range(target_ts.size(2))]
        mask_list   = [self._seperate(mask_ts[:, 0, 0])]
        output_list = [self._seperate(output_ts[:, 0, i]) for i in range(output_ts.size(2))] if output_ts is not None else None
        myrow = len(target_list)

        target_list= [j for i in target_list for j in i]
        mask_list= [j for i in mask_list for j in i]
        output_list= [j for i in output_list for j in i]

        bit_error = len([i for i, j in zip(target_list,output_list) if i!=j])
        total_bit = myrow * len([i for i in mask_list if i =='1'])
        avg_bit_error = bit_error/total_bit

        self.save_bit_error.append(bit_error)
        self.save_avg_bit_error.append(avg_bit_error)
        """
        print("bit error:\n",bit_error)
        print("avg bit error:\n",avg_bit_error)
        """
        #print("check if bit error is appending, length is now:",len(self.save_bit_error))

        if (len(self.save_bit_error) % 50 == 0):
            with open( 'BitE_' + self.refs + '.csv', "a") as myfile:
                myfile.write(str(self.save_bit_error[-1]))
                myfile.write("\n")
            with open( 'avgBitE_' + self.refs + '.csv', "a") as myfile:
                myfile.write(str(self.save_avg_bit_error[-1]))
                myfile.write("\n")

    def sample_random_action(self):
        pass

    def _generate_sequence(self):
        """
        generates [batch_size x num_items x len_word] data and
        prepare input & target & mask

        Returns:
        exp_state1[0] (input) : starts w/ a start bit, then the input seq
                              : then the output to be returned, then 0's
            [0 ... 0, 1, 0;   # start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 1, 0;   # 2nd start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 1, 0;   # 3rd start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 1, 0;   # 4th start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 1, 0;   # 5th start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 1, 0;   # 6th start bit
             data   , 0, 0;   # data with padded 0's             
             0 ... 0, 0, 1;   # end bit
             query  , 0, 0;   # query data (must not be the last data item)
             0 ... 0, 0, 1]   # end bit   
                          
        exp_state1[1] (target): 0's until after inputs has the end bit, then the
                              : seq to be copied, but w/o the extra channels for
                              : start and end bits
            [0 ... 0;         # (num_words+1)*num_items+1 rows of 0's
             data   ;         # data item that after the query item
             0 ... 0]         # a row of zeros
             
        exp_state1[2] (mask)  : 1's for all row corresponding to the target
                              : 0's otherwise}
            [0;               # (num_words+1)*num_items+1 rows of 0's
             1;               # num_words rows of 1's
             0];              # a row of zeros
        NOTE: we pad extra rows of 0's to the end of those batches with smaller
        NOTE: length to make sure each sample in one batch has the same length
        """
        self.exp_state1 = []
        # we prepare input, target, mask for each batch
        batch_num_words     = np.random.randint(self.min_num_words, self.max_num_words+1, size=(self.batch_size))
        max_batch_num_words = np.max(batch_num_words)

        batch_num_items = np.random.randint(self.min_num_items, self.max_num_items+1, size=(self.batch_size))
        max_batch_num_items = np.max(batch_num_items)

        self.exp_state1.append(np.zeros((self.batch_size, (max_batch_num_words + 1) * (max_batch_num_items + 2) + 1, self.len_word + 2))) # input
        self.exp_state1.append(np.zeros((self.batch_size, (max_batch_num_words + 1) * (max_batch_num_items + 2) + 1, self.len_word)))     # target
        self.exp_state1.append(np.zeros((self.batch_size, (max_batch_num_words + 1) * (max_batch_num_items + 2) + 1, 1))) # mask

        for batch_ind in range(self.batch_size):
            num_words = batch_num_words[batch_ind]
            #print(num_words)
            num_items = batch_num_items[batch_ind]
            #print(num_items)
            self.exp_state1[0][batch_ind][(num_words+1)*num_items][-1] = 1  # set an end bit for input items
            for j in range(num_items):
                data = np.random.randint(2, size=(num_words, self.len_word))
                # prepare input for this sample
                self.exp_state1[0][batch_ind][(num_words+1)*j][-2] = 1# set start bit
                self.exp_state1[0][batch_ind][(j*(num_words+1)+1) : (j*(num_words+1)+num_words+1), 0:-2] = data
            #copy ith item as query item
            i = np.random.choice(range(num_items-1)) 
            query_data = self.exp_state1[0][batch_ind][(i*(num_words+1)+1) : (i*(num_words+1)+num_words+1), 0:-2]
            self.exp_state1[0][batch_ind][((num_items)*(num_words+1)+1) : ((num_items)*(num_words+1)+1)+num_words, 0:-2] = query_data
            self.exp_state1[0][batch_ind][(num_items+1)*(num_words+1)][-1] = 1# set end bit for query items
            # prepare target for this sample
            target_data = self.exp_state1[0][batch_ind][((i+1)*(num_words+1)+1) : ((i+1)*(num_words+1)+num_words+1), 0:-2]
            #print(target_data)
            #print("im here")
            #temp = self.exp_state1[1][batch_ind][((num_items)*(num_words+1)+1) : ((num_items)*(num_words+1)+1+num_words), 0:]
            #print(temp)
            self.exp_state1[1][batch_ind][((num_items+1)*(num_words+1)+1) : ((num_items+1)*(num_words+1)+1+num_words), 0:] = target_data
            # prepare mask for this sample
            self.exp_state1[2][batch_ind][((num_items+1)*(num_words+1)+1) : ((num_items+1)*(num_words+1)+1)+num_words, 0] = 1

    def reset(self):
        self._reset_experience()
        self._generate_sequence()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self._generate_sequence()
        return self._get_experience()
