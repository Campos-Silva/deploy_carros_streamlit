import streamlit as st

import pandas as pd

from pycaret.classification import load_model, predict_model

modelo = load_model('meu-melhor-modelo-para-precos-veiculos')

paginas = ['Home',  'Precificação de Veículos Usados']

pagina = st.sidebar.radio('Navegue por aqui', paginas)

if pagina == "Home":
    st.title('Meus Modelos em Produção')

    st.write('Navegue pelo menu lateral para acessar todos os meus modelos em produção')

if pagina == "Precificação de Veículos Usados":
    st.title('Precificação de Veículos Usados')




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







