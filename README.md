# lstm-generator

## How To Use

1. Place your training data in the `data` directory. The training data file must be a text file, and the filename must end in `.txt`

2. Use `learn` to train the model. Specify the training data with the 
   `-m/--model` argument and do not include the file extension:
    ```shell
    ./learn --model oxford_dictionary
    ```   
    >***Note:** While the model is training, a log file of its progress (including test generated sequences) will be saved in the logs directory as `{model name}.log`.*
     
3. Use `generate` to generate text with the trained model:
    ```shell
    ./generate --model oxford_dictionary --seed looblap
    ```
              
For a full description of customizations check the help menus (`-h`) of both `learn` and `generate`.
