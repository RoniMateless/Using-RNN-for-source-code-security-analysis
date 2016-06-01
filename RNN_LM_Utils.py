import matplotlib.pyplot as plt

class SourceCodeUtils():

    @staticmethod
    def cut_to_semi_redundant_sequences(X_sequence, y_target, maxlen = 100, step = 100):
    # cut the text in semi-redundant sequences of maxlen characters
        sequences_cutted = []
        y_label_extended = []
        for seq in range(len(X_sequence)):
            seq_tail = len(X_sequence[seq]) - maxlen
            if seq_tail<=0:
                sequences_cutted.append(X_sequence[seq])
                y_label_extended.append(y_target[seq])
            else:
                for i in range(0, seq_tail, step):
                    sequences_cutted.append(X_sequence[seq][i: i + maxlen])
                    y_label_extended.append(y_target[seq])
        print('nb sequences:', len(sequences_cutted))

        #print 'Sample sequences:'
     #   for i in xrange(3):
#            print sequences_cutted[i]
        return sequences_cutted, y_label_extended

    @staticmethod
    def plot_loss(hist):
        acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']

        # plot loss, val loss
        nb_epoch = len(loss)
        plt.plot(range(nb_epoch), loss, marker='.', label='loss')
        plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()