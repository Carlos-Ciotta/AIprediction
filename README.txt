cosumption_model.h5{
    epochs: 2000
    eta: 1e-2
    batch_size: 32
    test_rate = 0.2

    GRU:{
        activation = 'relu'
        GRU LAYERS = 96 - 63 - 32 DENSE 1 DROPOUT 0.2
    }

    x = ['solarradiation'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal']
    y = ['Active Energy (MWh) - Porto']
}

cosumption_model2.h5{
    epochs: 2000
    eta: 1e-2
    batch_size: 32
    test_rate = 0.2

    GRU:{
        activation = 'silu'
        GRU LAYERS = 64 - 32 DENSE 1 DROPOUT 0.2
    }

    x = ['solarradiation'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal']
    y = ['Active Energy (MWh) - Porto']
}

cosumption_model4.h5{
    epochs: 600
    eta: 1e-2
    batch_size: 32
    test_rate = 0.2

    GRU:{
        activation = 'relu'
        GRU LAYERS = 64 - 32 DENSE 1
    }

    x = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal','Active Energy (MWh) - Porto']
    y = ['Active Energy (MWh) - Porto']
}
cosumption_model5.h5{
    epochs: 600
    eta: 1e-2
    batch_size: 32
    test_rate = 0.2

    GRU:{
        activation = 'hard_silu'
        GRU LAYERS = 64 - 32 DENSE 1
    }

    x = ['solarradiation','humidity','holiday'
                        ,'month','hour','Clear','Overcast','Partially cloudy',"Rain, Partially cloudy",
                        'Domingo','Quarta','Quinta','Segunda','Sexta','Sábado','Terça'
                        ,'Portugal','Active Energy (MWh) - Porto']
    y = ['Active Energy (MWh) - Porto']
}