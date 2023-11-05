This repository implements a module for recommending items to users. 

The **conf** directory contains configs pointing to file storage locations and IALS model configs.

The **item_rec** directory contains implementations of all the necessary modules for data processing, model training and testing.

The **notebooks** directory lies optionally.

**data** directory will be downloaded during _run.py_ program execution.

To run the program, just enter in the terminal: 
```console
python run.py
```

If you need to change a model parameter, for example, the number of iterations when training the IALS model, you can write the command as follows:
```console
python run.py ials_params.n_iter=30
```

After running run.py programm you will see Mean Average Precision score of random model and IALS model.
MAP@10 of IALS model is much greater than random one.