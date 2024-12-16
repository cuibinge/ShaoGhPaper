# ShaoGhPaper
A extraction method for raft laver culture area based on UNet and dual attention feature fusion.

Running environment as follows:

    python = 3.6.13
    pytorch = 1.4.0
    numpy = 1.19.2
    torch = 1.10.2
    tensorflow-gpu = 1.13.2
If you want to inpsect the completeness of our code, you could use the datasets that we have provided.And you need to merge imagesDataset_part_aa.npy,imagesDataset_part_ab.npy and imagesDataset_part_ac.npy  into the file called imagesDataset_.npy while replace the original imagesDataset_.npy.After that,you can run the `main.py` successfully.

In general, I sure you don't just want to run our code.And now,let me tell you how to modify our code if you will use your own datasets.
* Firstly, you need to delete all our files with the suffix `.npy` and replace them with your `.npy`.
* Secondly, you need to modify the path in the `readNpy()` function in the `data.py` file at the same time you need to pay attention to lines 23 to 25 in the `main.py` as well as lines 32 and 33 in the `test.py`, these all the code you need to modify.
* To the end, you need to pay attention to the lines 51 to 72 in the `main.py`.You can look at the efforts that I do it for data format, I think you will do the same thing like me for your own dataset.Actually, it is not difficult, you just need to use `resize()` function reasonably.

Finishing all the above works, you will run your code successfully.
