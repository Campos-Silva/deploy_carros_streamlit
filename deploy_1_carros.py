import streamlit as st

import pandas as pd

from pycaret.classification import load_model, predict_model

from PIL import Image


modelo = load_model('meu-melhor-modelo-para-precos-veiculos')

paginas = ['Home',  'Precificação de Veículos Usados']

pagina = st.sidebar.radio('Navegue por aqui', paginas)

if pagina == "Home":
    st.title('Meus Modelos em Produção')

    st.write('Navegue pelo menu lateral para acessar todos os meus modelos em produção')







#________________________________



if pagina == "Precificação de Veículos Usados":
    st.title('Precificação de veículos usados')


    st.markdown('Essa página contêm o modelo de "Precificação de Veículos Usados" em *produção*. O construí para ajudar pessoas, concessionárias e distribuidoras a obterem uma estimativa de preços de veículos baseados em suas caracteristicas.')
    
    st.markdown('Para detalhes da construção desse modelo acesse [*aqui*](https://github.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados). Ali você econtrará informações sobre o [*conjunto de dados base*](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho), e as etapas de [*Pré-Processamento e Limpeza*](https://github.com/Campos-Silva/Projeto_01_Parte_A_Importacao-e-limpeza-de-dados-no-Python), [*Exploração dos Dados*](https://github.com/Campos-Silva/Projeto_01_Parte_B_Exploracao_de_dados_no_Python) e [*Construção do Modelo Preditivo*](https://github.com/Campos-Silva/Projeto_01_Parte_C_Modelos_de_Machine_Learning_no_Python)')

    st.markdown('### **Contextualização - Venda de veículos usados**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados/main/preco_carro_a.png")

    st.markdown('Quando vamos comprar um **carro usado** uma das primeiras questões que devemos checar é **se o preço está de acordo com as características do veículo**. Esse tipo de pensamento deve valer também ao vendedor. Agências que compram veículos usados ou seminovos para revender precisam comprá-los com preços condizentes as suas características, isso para que possam ter lucro nessas revendas. Assim, saber quanto vale um veículo não é uma questão trivial, e sim uma questão que influencia nas nossas tomadas de decisão (ENGERS; HARTMANN; STERN, 2009; OU et al., 2020).')

    st.markdown('### **Setor em crescimento no Brasil**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados/main/Apresenta%C3%A7%C3%A3o1_b.png")

    st.markdown('Essa questão é ainda mais importante nos dias de hoje, uma vez que o setor de negócios de compra e venda de veículos está em alta. No Brasil, a venda de carros usados e seminovos tiveram cerca de 8% a mais de vendas acumuladas quando comparadas com o mesmo período de 2019, período anterior a pandemia (FOLHA, 2021 , FENAUTO, 2021). Já os primeiros sete meses de 2021 tiveram acréscimo de vendas acumuladas de 55%, em comparação com o mesmo período de 2020 (FOLHA, 2021 ). Esse cenário se deve, principalmente, ao fato de que a pandemia do COVID-19 tem impactado negativamente o fluxo de produção das montadoras, o que ocasionou em 2020 a que queda de 32% de venda de carros novos (El PAÍS, 2021, FENAUTO, 2021).')

    st.markdown('### **Importância **')

    st.markdown('Esse modelo permite a tomada de decisões visando a maximização dos lucros e redução de perdas por meio da precificação adequada de veículos baseados em suas características.')

    st.markdown('---')


    st.markdown('### **Como usar o Modelo?**')

    st.markdown('Utilize as variáveis abaixo para gerar o preço previsto do veículo baseado nessas características')



    st.markdown('---')




    kilometragem = st.slider('Kilometragem',
                     min_value = 1000,
                     max_value = 195000,
                     value = 65000,
                     step = 500)


    tipo_combustivel = st.selectbox('Combustível',['Gasolina','Diesel'])


    tipo_do_vendedor = st.selectbox('Tipo do Vendedor',['Individual','Distribuidora', 'Concessionaria'])


    transmissao = st.selectbox('Transmissão',['Manual','Automatico'])


    dono = st.selectbox('Quantidade de Donos',['1º dono','2º dono', '3º dono', '4º dono e acima'])


    consumo_do_combustivel_kmpl = st.slider('Consumo do combustível (Km/L)',
                     min_value = 12,
                     max_value = 28,
                     value = 15,
                     step = 1)


    motor_CC = st.slider('Cilindradas do Motor (CC)',
                     min_value = 950,
                     max_value = 1550,
                     value = 1260,
                     step = 50)


    potencia_do_motor_bhp = st.slider('Potência do Motor (BHP)',
                     min_value = 48,
                     max_value = 115,
                     value = 80,
                     step = 5)

    Marca = st.selectbox('Marca',['Chevrolet','Datsun','Fiat','Ford','Honda','Hyundai','Mahindra','Maruti','Nissan','Renault','Skoda','Tata','Toyota','Volkswagen'])


    dados0 = {'kilometragem': [kilometragem],
              'tipo_combustivel': [tipo_combustivel],
              'tipo_do_vendedor': [tipo_do_vendedor],
              'transmissao': [transmissao],
              'dono': [dono],
              'consumo_do_combustivel_kmpl': [consumo_do_combustivel_kmpl],
              'motor_CC': [motor_CC],
              'potencia_do_motor_bhp': [consumo_do_combustivel_kmpl],
              'Marca': [Marca]}

    dados = pd.DataFrame(dados0)

    st.markdown('---')

    if st.button('EXECUTAR MODELO'):
        pred = float(predict_model(modelo, data = dados)['Label'].round(2)*6)
        saida = 'O valor predito é de R${:.2f}'.format(pred)
        st.subheader(saida)




