# SCA — Sistema de Confiabilidade de Ativos

Este projeto começou enquanto eu estudava confiabilidade e análise estatística aplicada à manutenção. Eu queria ir além de simplesmente colocar os dados em uma ferramenta pronta e receber o resultado: queria entender melhor o que estava sendo calculado e como interpretar cada indicador.

Como não sou desenvolvedora, fui construindo a aplicação em Python com apoio de IA, estudando os conceitos estatísticos e testando cada etapa. O projeto ainda está em evolução e também funciona como meu ambiente de aprendizado.

## O que a aplicação analisa

- dados de vida com falhas e censuras à direita;
- curva de Kaplan-Meier;
- ajuste de distribuições por máxima verossimilhança;
- comparação dos modelos por AIC, BIC, ΔAIC e Peso de Akaike;
- distribuição Weibull 2P e interpretação do parâmetro β;
- R(t), F(t), MTTF, B10, B50 e B90;
- geração de relatório em PDF por distribuição;
- separação conceitual entre análise de vida e MTBF/confiabilidade operacional.

Cada linha da base representa uma **observação de vida**, e não necessariamente um equipamento diferente.

## Por que mantive este projeto no portfólio

Além do resultado final, o que mais me interessa neste projeto é o processo de aprendizado. Construir as análises me obrigou a entender melhor de onde vêm os números, o que cada cálculo representa e quais cuidados são necessários antes de interpretar um resultado de confiabilidade.

Ele também representa uma área que tenho buscado desenvolver: juntar meu conhecimento de manutenção com estatística, dados, programação e ferramentas de IA para resolver problemas de forma mais estruturada.

## Sobre esta versão

Esta é uma versão demonstrativa para estudo e portfólio. Ela não acessa a base operacional utilizada no ambiente de trabalho e não grava dados em banco.