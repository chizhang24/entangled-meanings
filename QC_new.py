import pennylane as qml
from pennylane import numpy as np

from cin.dataset             import iris, pima, haberman, banknote_authentication, manipulate, lambeq_data_loader , amazon_data_loader
from cin.pennylane.common    import numpy_interface
from cin.pennylane.templates import qubit_hqc, amp_hqc, dc_hqc


def main(X, Y, encoding:str, reps, steps):
    try:
        # config.
        # reps = 2 # repetições
        # steps = 20   # iterações
        lr = 0.1      # learning rate
        batch = 0.1   # batch set size
        val = 0.2     # validation set size
        test = 0.2    # test set size
        verbose = 1   # 0: just final results; 1: prints every iteration

        # template (select one).
        #tplt = qubit_hqc                  # qubit encoding              + hierarchical quantum classifier
        #tplt = amp_hqc                    # amplitude encoding          + hierarchical quantum classifier
        #tplt = dc_hqc                     # divide-and-conquer encoding + hierarchical quantum classifier
            
        # dataset (select one).
        #X, Y = iris.load([0, 1])
        #X, Y = pima.load()
        #X, Y = banknote_authentication.load()
        #X, Y = haberman.load()
        #X, Y = lambeq_data_loader.load()
        # X, Y = amazon_data_loader.wv_load()

        
        if encoding == 'amplitude':
            tplt = amp_hqc
        if encoding == 'dc':
            tplt = dc_hqc



        # parameterized variational circuit.
        circuit = tplt.circuit

        # circuit properties.
        n, N, w, X = tplt.config(X)
        print("n={}, N={}, w={}, x={}".format(n, N, w, len(X)))

        # load a Pennylane plugin device and return the instance.
        dev1 = qml.device("default.qubit", wires=N)

        # splits the data, "reps" times.
        splitted = [manipulate.split_data(X, Y, validation_size=val, test_size=test) for i in range(reps)]
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
