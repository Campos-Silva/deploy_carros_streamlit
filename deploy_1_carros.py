import streamlit as st

import pandas as pd

from pycaret.classification import load_model, predict_model

from PIL import Image


modelo = load_model('meu-melhor-modelo-para-precos-veiculos')

modelo_custos_seguro = load_model('modelo-para-custos-seguro')

modelo_para_deteccao_de_fumantes = load_model('modelo-para-deteccao-de-fumantes')


paginas = ['Home',  'Precificação de Veículos Usados', 'Custos de Seguro de Saúde', 'Detecção de Fraudadores de Seguro']

pagina = st.sidebar.radio('Navegue por aqui', paginas)


if pagina == "Home":
    st.title('Meus Modelos em Produção')

    st.write('Navegue pelo menu lateral para acessar os meus modelos em produção')


#________________________________



if pagina == "Precificação de Veículos Usados":
    st.title('Precificação de veículos usados')

    st.image('https://raw.githubusercontent.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados/main/preco_carro_a.png', width=500)

    st.markdown('Construí esse modelo preditivo para ajudar pessoas, concessionárias e distribuidoras a obterem uma estimativa de preços de veículos baseados em suas características.')

    st.markdown('### **Como usar o Modelo?**')

    st.markdown('Configure as características do veículo abaixo com o valor desejado e depois clique em "EXECUTAR MODELO" para gerar o seu preço previsto.')

    st.markdown('---')

    st.markdown('### **Características do veículo**')


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
        saida = 'O preço estimado do veículo é de R${:.2f}'.format(pred)
        st.subheader(saida)
        st.markdown('')
        st.markdown('Muito obrigado! Espero que tenha gostado 😄✌️')
        st.markdown('Caso deseje mais informações sobre Data Science me siga no [LinkedIn](https://www.linkedin.com/in/lucas-andrei-campos-silva/)')


    st.markdown('---')

    st.markdown('# **Importância do modelo**')

    st.markdown('### **Área de Negócios: Venda de veículos usados**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados/main/comprar_carro_duvida_c.png")

    st.markdown('Quando vamos comprar um **carro usado** uma das primeiras questões que devemos checar é **se o preço está adequado as características do veículo**. Essa questão deve valer também ao vendedor. Agências que compram veículos usados para revender precisam comprá-los com preços condizentes as suas características, isso para que possam ter lucro nessas revendas. Assim, saber quanto vale um veículo não é uma questão trivial, e sim uma questão que influencia nas nossas tomadas de decisão (ENGERS; HARTMANN; STERN, 2009; OU et al., 2020).')

    st.markdown('### **Setor em crescimento no Brasil**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados/main/Apresenta%C3%A7%C3%A3o1_b.png")

    st.markdown('Essa questão é ainda mais importante nos dias de hoje, uma vez que esse setor de negócios está em alta. No Brasil, a venda de carros usados e seminovos tiveram cerca de 8% a mais de vendas acumuladas quando comparadas com o mesmo período de 2019, período anterior a pandemia (FOLHA, 2021 , FENAUTO, 2021). Já os primeiros sete meses de 2021 tiveram acréscimo de vendas acumuladas de 55%, em comparação com o mesmo período de 2020 (FOLHA, 2021 ). Esse cenário se deve, principalmente, ao fato de que a pandemia do COVID-19 tem impactado negativamente o fluxo de produção das montadoras, o que ocasionou em 2020 a que queda de 32% de venda de carros novos (El PAÍS, 2021, FENAUTO, 2021).')

    st.markdown('### **Importância**')

    st.markdown('Esse modelo permite a tomada de decisões visando a maximização dos lucros e redução de perdas por meio da precificação adequada de veículos baseados em suas características.')

    st.markdown('Para detalhes sobre a sua construção do modelo acesse [*aqui*](https://github.com/Campos-Silva/Projeto_1_Precificacao_de_Veiculos_Usados). No link você econtrará informações sobre o [*conjunto de dados base*](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho), e as etapas de [*Pré-Processamento e Limpeza*](https://github.com/Campos-Silva/Projeto_01_Parte_A_Importacao-e-limpeza-de-dados-no-Python), [*Exploração dos Dados*](https://github.com/Campos-Silva/Projeto_01_Parte_B_Exploracao_de_dados_no_Python) e [*Construção do Modelo Preditivo*](https://github.com/Campos-Silva/Projeto_01_Parte_C_Modelos_de_Machine_Learning_no_Python).')

    st.markdown('---')


    col1, col2 = st.columns([0.3, 1])

    with col1:
        st.markdown("### Autor")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.image('https://raw.githubusercontent.com/Campos-Silva/Campos-Silva/main/perfil_lucas_andrei_campos_silva.png', width=150)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Lucas Andrei Campos-Silva**")
        st.markdown("Data Scientist | Business Intelligence (BI) | Analista de Dados | Ph.D. Student")
        st.markdown("")
        st.markdown("Contatos")
        st.markdown("[![linkedin](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/linkedin_v_35.png)](https://www.linkedin.com/in/lucas-andrei-campos-silva/)")
        st.markdown("[![github](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/github_v_50.png)](https://github.com/Campos-Silva)")



if pagina == 'Custos de Seguro de Saúde':


    st.title('Modelo para Previsão de Custos de Seguro de Saúde')

    st.image('https://raw.githubusercontent.com/Campos-Silva/custos_seguro/main/imagem_2.jpg', width=400)

    st.markdown('Construí esse modelo preditivo para ajudar Agências de Seguro de Saúde a obterem uma estimativa de preços de seguros de seus clientes baseados em suas características.')

    st.markdown('### **Como usar o Modelo?**')

    st.markdown('Insira as informações do cliente com o valor desejado e depois clique em "EXECUTAR MODELO" para gerar o seu preço previsto.')

    st.markdown('---')

    st.markdown('### **Informações do cliente**')

    Idade = st.slider('Idade',
                     min_value = 18,
                     max_value = 65,
                     value = 30,
                     step = 1)

    Sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])

    IMC = st.slider('IMC',
                     min_value = 15,
                     max_value = 54,
                     value = 24,
                     step = 1)


    Filhos = st.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])

    Fumante = st.selectbox("O cliente é fumante?", ['Sim', 'Nao'])

    Regiao = st.selectbox("Região em que reside", 
                                      ['Sudeste', 'Sudoeste', 'Nordeste', 'Noroeste'])


    dados0 = {'Idade': [Idade], 'Sexo': [Sexo], 'IMC': [IMC], 'Filhos': [Filhos], 'Fumante': [Fumante], 'Regiao': [Regiao]}
    dados = pd.DataFrame(dados0)

    st.markdown('---')

    if st.button('EXECUTAR MODELO'):
        pred = float(predict_model(modelo_custos_seguro, data = dados)['Label'].round(2))
        saida = 'O valor do seguro é de  ${:.2f}'.format(pred)
        st.subheader(saida)
        st.markdown('')
        st.markdown('Muito obrigado! Espero que tenha gostado 😄✌️')
        st.markdown('Caso deseje mais informações sobre Data Science me siga no [LinkedIn](https://www.linkedin.com/in/lucas-andrei-campos-silva/)')

    st.markdown('---')

    st.markdown('# **Importância do modelo**')

    st.markdown('### **Área de Negócios: Seguro de Saúde**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/custos_seguro/main/imagem_3.jpg")

    st.markdown('O setor de mercado de seguros tem a missão de proteger seus clientes contra riscos de perdas de seus patrimônios e assegurar proteção à vida e a saúde para as famílias (CNSEG, 2021).  Como o objeto desse segmento de seguros é a "saúde", certas características do cliente que podem direcionar o seu quadro clínico e o de seus familiares são levados em consideração na precificação do seguro. Isso para que não haja prejudicados durante o acordo de seguro, uma vez que esse setor preza pela boa fé e a solidariedade, consideradas como base para o mutualismo entre seguradora e cliente (Fenasaude, 2018; CNSEG, 2021).')

    st.markdown('### **Características do cliente e o preço do seguro**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/custos_seguro/main/raio_x.jpg")

    st.markdown('Certas características de uma pessoa podem leva-lá a desenvolver algum tipo de doença, seja ela fatal ou não. Características como hábito de fumar, alto Índice de Massa Corpórea (IMC), e aumento da idade são três exemplos. É conhecido que o tabagismo gera no mínimo 50 tipos de enfermidades, como doenças respiratórias e cardiovasculares e câncer (INCA, 2022). Já relacionado ao IMC, a obesidade pode acarretar em prejuízos a saúde do indivíduo como dificuldades respiratórias e  doenças cardiovasculares (Pinheiro, 2004). Outra característica que pode favorecer a presença de doenças é o aumento da idade, favorecendo o surgimento de doenças crônicas não transmissíveis, como doenças respiratórias, cardiocirculatórias e diabetes (Figueiredo et al., 2021).')

    st.markdown('Especificamente relacionado a essas três características, as seguradoras de saúde tendem dar maior valor do seguro à pessoas que as possuem. Isso porque elas podem levar uma pessoa a precisar mais do aparato do sistema de saúde do que uma pessoa que não as apresenta.')

    st.markdown('### **Importância**')

    st.markdown('Esse modelo permite a tomada de decisões visando a maximização dos lucros e redução de perdas por meio da precificação adequada do custo do seguro para o cliente.')

    st.markdown('---')


    col1, col2 = st.columns([0.3, 1])

    with col1:
        st.markdown("### Autor")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.image('https://raw.githubusercontent.com/Campos-Silva/Campos-Silva/main/perfil_lucas_andrei_campos_silva.png', width=150)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Lucas Andrei Campos-Silva**")
        st.markdown("Data Scientist | Business Intelligence (BI) | Analista de Dados | Ph.D. Student")
        st.markdown("")
        st.markdown("Contatos")
        st.markdown("[![linkedin](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/linkedin_v_35.png)](https://www.linkedin.com/in/lucas-andrei-campos-silva/)")
        st.markdown("[![github](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/github_v_50.png)](https://github.com/Campos-Silva)")



if pagina == 'Detecção de Fraudadores de Seguro':
    st.title('Modelo para Detecção de Fraudadores de Seguro de Saúde')


    st.image('https://raw.githubusercontent.com/Campos-Silva/custos_seguro/main/imagem_fraude.png', width=300)

    st.markdown('Construí esse modelo preditivo para ajudar Agências de Seguro de Saúde a detectarem clientes que estejam possivelmente fraudando informações e assim, obtendo valor de seguro abaixo da tabela de preços. Isso pode acarretar graves prejuízos econômicos a seguradora.')

    st.markdown('A característica a ser detectada é se o cliente é de fato fumante ou não.')

    st.markdown('### **Como usar o Modelo?**')

    st.markdown('Insira as informações do cliente com o valor desejado e depois clique em "EXECUTAR MODELO" para detectar se o cliente é fumante ou não.')

    st.markdown('---')

    st.markdown('### **Informações do cliente**')


    Idade = st.slider('Idade',
                     min_value = 18,
                     max_value = 65,
                     value = 30,
                     step = 1)

    Sexo = st.selectbox('Sexo', ['Masculino', 'Feminino'])
    
    IMC = st.slider('IMC',
                     min_value = 15,
                     max_value = 54,
                     value = 24,
                     step = 1)

    Filhos = st.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])
    Custos = st.slider('Custos', 1000, 50000, 10000, 1000)
    Regiao = st.selectbox("Região em que mora", 
                                      ['Sudeste', 'Sudoeste', 'Nordeste', 'Noroeste'])


    dados0 = {'Idade': [Idade], 'Sexo': [Sexo], 'IMC': [IMC], 'Filhos': [Filhos], 'Custos': [Custos], 'Regiao': [Regiao]}
    dados = pd.DataFrame(dados0)


    dados = pd.DataFrame(dados0)

    st.markdown('---')

    if st.button('EXECUTAR MODELO'):
        pred = predict_model(modelo_para_deteccao_de_fumantes, data = dados)

        resp = 'NÃO' if pred['Label'][0] == 'Nao' else 'SIM' 
        
        prob = pred['Score'][0]
        


        st.write('O cliente é fumante: **{}, com probabilidade de {:.2f}%**'.format(resp, 100*prob))
        st.markdown('')
        st.markdown('Muito obrigado! Espero que tenha gostado 😄✌️')
        st.markdown('Caso deseje mais informações sobre Data Science me siga no [LinkedIn](https://www.linkedin.com/in/lucas-andrei-campos-silva/)')
        


    st.markdown('---')


    st.markdown('# **Importância do modelo**')

    st.markdown('### **Área de Negócios: Seguro de Saúde: Detecção de Fraudes**')

    st.image("https://raw.githubusercontent.com/Campos-Silva/custos_seguro/main/imagem_5.png")

    st.markdown('Sabendo que certas características, como alto Índice de Massa Corpórea, hábito de fumar e aumento da idade, podem elevar o preço do seguro de saúde, certas pessoas podem usar de má fé e mentir durante o preenchimento de suas informações na hora da assinatura do contrato. Essa ação, é caracterizada como fraude e causa significativas perdas econômicas ao setor de seguros (SQF, 2020). Com o objetivo de monitorar possíveis perdas econômicas em decorrência de fraudes relacionadas ao setor de mercado de seguros, o Brasil vem monitorando e quantificando o quanto é perdido. Essas cifras ficam em torno dos milhões de reais. Por exemplo, no ano de 2020, foi comprovada a perda de quase 330 milhões de reais no primeiro semestre e 391 milhoes de reais no segundo semestre (SQF, 2020). Como é perdido muito dinheiro devido a fraudes, detectá-las é uma questão de alta relevância a esse setor de mercado.')

    st.markdown('### **Importância**')

    st.markdown('Esse modelo permite a tomada de decisões visando a redução de perdas financeiras por meio da detecção de possíveis fraudadores do seguro.')

    st.markdown('---')


    col1, col2 = st.columns([0.3, 1])

    with col1:
        st.markdown("### Autor")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.image('https://raw.githubusercontent.com/Campos-Silva/Campos-Silva/main/perfil_lucas_andrei_campos_silva.png', width=150)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("#### **Lucas Andrei Campos-Silva**")
        st.markdown("Data Scientist | Business Intelligence (BI) | Analista de Dados | Ph.D. Student")
        st.markdown("")
        st.markdown("Contatos")
        st.markdown("[![linkedin](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/linkedin_v_35.png)](https://www.linkedin.com/in/lucas-andrei-campos-silva/)")
        st.markdown("[![github](https://raw.githubusercontent.com/Campos-Silva/deploy_carros_streamlit/main/github_v_50.png)](https://github.com/Campos-Silva)")