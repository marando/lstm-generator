# lstm-generator

## How To Use

1. Place your training data in the `./data` directory
    >***Note:** The training data file must be text format, and the filename 
                must end in .txt*

2. Use `./learn` to train the model. Specify the training data with the 
   `-m/--model` argument.
    ```shell
    ./learn --model oxford_dictionary
    ```
    >***Note:** Do not include .txt in the argument.*
     
3. Use `./generate` to generate text with the trained model:
    ```shell
    ./generate --model oxford_dictionary --seed looblap
    ```
     While the model is training, a log file of its progress including test 
     generated sequences will be saved in the logs directory as 
     `{model name}.log`.
       
A full description of customizations is available in the in the help menus `-h` for 
both `./learn` and `./generate` 