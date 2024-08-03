import pennylane as qml
from pennylane import numpy as np
from QuantumEncodings import amp_enc, dc_enc
import time 

def split_data(X, Y, validation_size=0.1, test_size=0.1):
    np.random.seed(0) 
    num_data = len(Y)
    num_test = int(test_size * num_data)
    num_val = int(validation_size * num_data)
    
    index = np.random.permutation(range(num_data)) 
    
    X_test = X[index[:num_test]]
    Y_test = Y[index[:num_test]]
    X_val = X[index[num_test:num_val+num_test]]
    Y_val = Y[index[num_test:num_val+num_test]]
    X_train = X[index[num_val+num_test:]]
    Y_train = Y[index[num_val+num_test:]]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test




def variational_classifier(node, var, state_vector=None, n=4):
    weights = var[0:-1]
    bias = var[-1]
    return node(weights, state_vector=state_vector, n=n) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + ((l - p)/2) ** 2

    loss = loss / len(labels)
    return loss

def cost(node, n, var, state_vectors, labels):
    predictions = [variational_classifier(node, var, state_vector=state_vector, n=n) for state_vector in state_vectors]
    return square_loss(labels, predictions)

def accuracy(labels, predictions):
    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc

def numpy_interface(init_node, init_var, n, lr, steps, batch_size, X_train, X_val, X_test, Y_train, Y_val, Y_test, show=False):
    if (show):
        print('\nNumPy interface:')

    X = [*X_train, *X_val]
    Y = [*Y_train, *Y_val]

    best = [0, 1.0, 0.0, 0.0, 0.0, []]

    node = init_node
    var = init_var
    opt = qml.optimize.AdamOptimizer(stepsize=lr)

    time1 = time.time()
    for it in range(steps):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,)) # pylint: disable=no-member
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(node, n, v, X_train_batch, Y_train_batch), var)

        # Compute predictions on train and validation set
        predictions_train = [2*(variational_classifier(node, var, state_vector=f, n=n)>0.0)-1 for f in X_train] 
        predictions_val = [2*(variational_classifier(node, var, state_vector=f, n=n)>0.0)-1 for f in X_val]     
        
        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        # Compute cost on complete dataset
        cost_set = cost(node, n, var, X, Y)

        #if (cost_set < best[1]):
        if (acc_val > best[3] or (acc_val == best[3] and cost_set < best[1])):
          best[0] = it + 1
          best[1] = cost_set
          best[2] = acc_train
          best[3] = acc_val
          best[4] = 0.0
          best[5] = var

        if (show):
            print(
                "Iter:{:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
                "".format(it + 1, cost_set, acc_train, acc_val)
            )
    
    # Compute predictions on test set
    predictions_test = [2*(variational_classifier(node, best[5], state_vector=f, n=n)>0.0)-1 for f in X_test]
    # Compute accuracy on test set
    acc_test = accuracy(Y_test, predictions_test)
    best[4] = acc_test

    time2 = time.time()

    if (show):
        print("Optimized rotation angles: {}".format(best[5][:-1]))
        print("Optimized bias: {}".format(best[5][-1]))
        print("Optimized test accuracy: {:0.7f}".format(acc_test))
        print(f'Run time={((time2-time1)*1000.0):.3f}')

    return best




















def main(X, Y, encoding:str, reps, steps):
    try:
        lr = 0.1      # learning rate
        batch = 0.1   # batch set size
        val = 0.2     # validation set size
        test = 0.2    # test set size
        verbose = 1   # 0: just final results; 1: prints every iteration

       
        
        if encoding == 'amplitude':
            tplt = amp_enc
        if encoding == 'dc':
            tplt = dc_enc



        # parameterized variational circuit.
        circuit = tplt.circuit

        # circuit properties.
        n, N, w, X = tplt.config(X)
        print("n={}, N={}, w={}, x={}".format(n, N, w, len(X)))

        # load a Pennylane plugin device and return the instance.
        dev1 = qml.device("default.qubit", wires=N)

        # splits the data, "reps" times.
        splitted = [split_data(X, Y, validation_size=val, test_size=test) for i in range(reps)]
        batch_size = int(batch * len(splitted[0][0]))
        print("train={}, val={}, test={}, batch={}".format(len(splitted[0][0]), len(splitted[0][2]),len(splitted[0][4]), batch_size))
        
        # converts "circuit" it into a QNode instance.
        node = qml.QNode(circuit, dev1)

        # randomizes the init weights, "reps" times.
        vars = [np.array([*np.random.uniform(low=-np.pi, high=np.pi, size=w), 0.0]) for i in range(reps)] # pylint: disable=no-member

        # learn and test, "reps" times.
        best_iters = [numpy_interface(node, vars[i], n, lr, steps, batch_size, split[0], split[2], split[4], split[1], split[3], split[5], verbose) for i, split in enumerate(splitted)]

        # print best results.
        print('\nBest iters:')
        for best in best_iters:
            print(
                "Iter:{:5d} | Cost:{:0.7f} | Ac train:{:0.7f} | Ac val:{:0.7f} | Ac test:{:0.7f}"
                "".format(best[0], best[1], best[2], best[3], best[4])
            )
        
    
        print('\nBest bias:')
        for best in best_iters:
            print("Optimized bias: {}".format(best[5][-1]))



        return [n, best[2], best[3], best[4], best[5][-1]]
    
    except Exception as e:
        print(f"Error in classifier: {str(e)}")
        return None


# if __name__ == "__main__":
#    # This part will only run if the script is executed directly
#     X, Y = amazon_data_loader.wv_load()  # Load data here for direct execution
#     main(encoding=amp_hqc, X=X, Y=Y)
