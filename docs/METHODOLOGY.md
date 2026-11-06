# Metodologia numérica e proveniência

Este simulador usa um único ativo agregado com movimento browniano geométrico
(GBM). Os resultados são condicionais aos parâmetros, não previsões de WEGE3,
IBOV ou de uma carteira diversificada. Revisão numérica: 06/11/2026.

## Modelo e estatísticas

Para horizonte `T`, são usados `ceil(12*T)` intervalos iguais, com
`dt = T/ceil(12*T)`. A grade chega exatamente a `T`, inclusive quando o horizonte
não é um número inteiro de meses. Não é um calendário de negociação.

```text
log(V[t+dt]/V[t]) = (mu - sigma²/2)*dt + sigma*sqrt(dt)*Z
Z ~ N(0, 1), independente entre caminhos e intervalos
```

São transições exatas do GBM, sem discretização de Euler. `mu` é o drift contínuo
aritmético: `E[V(T)] = V(0)*exp(mu*T)`. Não é CAGR nem o retorno simples esperado
de um ano, que é `exp(mu)-1`. A mediana é `V(0)*exp((mu-sigma²/2)*T)`; o quantil
de ordem `p` inclui `sigma*sqrt(T)*Phi^-1(p)` no expoente.

Os percentis empíricos usam interpolação linear do NumPy. A probabilidade de
perda conta estritamente `V(T) < V(0)`: empates não são perdas. Com volatilidade
zero, os caminhos seguem `V(0)*exp(mu*t)`; com drift também zero, ficam exatamente
no saldo inicial, evitando perdas falsas por arredondamento de `exp(log(V(0)))`.
A probabilidade analítica usa `erfc` para preservar caudas pequenas. Valores
monetários analíticos que zeram por underflow, ou deixam de ser finitos, produzem
`ValueError`; o simulador também rejeita estouros numéricos. Não há precisão
arbitrária para parâmetros extremos.

P5 e P95 delimitam resultados do modelo, não um intervalo de confiança para a
estimativa. Com 320 caminhos, cada perda altera a frequência em 0,3125 ponto
percentual. O erro padrão dessa frequência, sob o modelo, é
`sqrt(p*(1-p)/320)` (até aproximadamente 2,8 pontos percentuais).
Isso não inclui incerteza nos parâmetros ou erro de especificação do modelo.

## Seed e reprodução

Cada execução cria seu próprio `numpy.random.default_rng(seed)` e não usa o
estado aleatório global. A mesma configuração no mesmo ambiente reproduz os
caminhos. Alterar horizonte muda a dimensão dos sorteios; a seed não define uma
trajetória independente da grade. Guarde configuração, versões das dependências
e dados de entrada para reproduzir um estudo.

Não há garantia de igualdade bit a bit entre versões, builds e máquinas distintas;
essa restrição segue a [política de compatibilidade do NumPy](https://numpy.org/doc/stable/reference/random/compatibility.html).
Os testes combinam igualdade de execuções repetidas, transições fechadas numa
grade fracionária e 50.000 caminhos confrontados com momentos e quantis
analíticos. Uma seed fixa evita testes estatísticos que mudem a cada execução.

## Calibração histórica opcional

`--ticker WEGE3` é normalizado para `WEGE3.SA`; `--ibov` solicita `^BVSP`.
O adaptador pede cinco anos de observações diárias ao Yahoo Finance por
`yfinance.download`, com `auto_adjust=True` e colunas simples. Usa a coluna
`Close` ajustada pelo fornecedor, sem aplicar outro ajuste. Os parâmetros da
consulta seguem a [documentação do yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

```text
r[i] = log(close[i]) - log(close[i-1])
sigma = sample_std(r, ddof=1) * sqrt(252)
mu = mean(r)*252 + sigma²/2
```

São exigidos pelo menos 31 preços (30 retornos), positivos e finitos, com índice
único e crescente. Esse mínimo é apenas uma proteção de entrada, não evidência
de suficiência estatística. O estimador usa variância amostral, não o estimador
de máxima verossimilhança com divisor `n`. A correção `sigma²/2` converte a média
logarítmica no drift do GBM.

252 é uma convenção de anualização. Cada par adjacente vale um intervalo diário;
o código não verifica calendário B3 nem repondera lacunas de pregões. Não ordena,
preenche ou remove dados inválidos dentro de `calibrate`. O fornecedor pode omitir
linhas ausentes antes de retorná-las. Séries de frequências diferentes não devem
ser passadas como preços diários. `^BVSP` é uma série de índice, não o resultado
líquido de investir num fundo que o acompanha.

A proveniência registra fornecedor, símbolo normalizado, horário UTC da coleta,
primeira/última observação, número de retornos e fórmula. A janela de cinco anos
é móvel; o fornecedor pode revisar ajustes e o último fechamento pode ainda ser
parcial. Não há arquivo histórico point-in-time nem cache persistente de preços
no pacote. A proveniência sozinha não recompõe os preços de uma coleta passada.
Falhas de download ou dados vazios geram erro: dados sintéticos só entram com
`--offline` explícito.

Smoke de integração em 06/11/2026, yfinance 1.8.0, sem incluir os preços no repo:

| Símbolo | Coleta UTC | Preços inicial/final | Retornos | mu | sigma |
| --- | --- | --- | ---: | ---: | ---: |
| WEGE3.SA | 21:40:30 | 08/11/2021–06/11/2026 | 1.247 | 0,1215890730 | 0,3034181933 |
| ^BVSP | 21:40:31 | 08/11/2021–06/11/2026 | 1.247 | 0,1136787086 | 0,1880287025 |

Esses números registram uma verificação do adaptador; não são premissas
recomendadas, fixtures de CI ou comprovação de qualidade preditiva.

## Fixture offline e CI

[`synthetic_daily.csv`](../src/portfolio_sim/fixtures/synthetic_daily.csv) foi
gerada por [`scripts/generate_fixture.py`](../scripts/generate_fixture.py), com
metadados em [`provenance.json`](../src/portfolio_sim/fixtures/provenance.json).
Não contém preços históricos de WEGE3 nem de IBOV.

- Geração original: 06/11/2026; seed `20261106`; drift 0,12; volatilidade 0,25.
- 757 preços, 756 retornos, de 03/01/2022 a 26/11/2024; calendário de segunda a
  sexta, incluindo possíveis feriados da bolsa; preços gravados com dez decimais.
- Calibração da amostra: `mu = 0.08664353968956609`,
  `sigma = 0.25040837685909034`. A diferença para os parâmetros geradores é
  esperada numa amostra finita, especialmente para o drift.
- SHA-256 do CSV: `aa9fc3a58f63fa85e9a012c486e765811939c38f420c80111726491b0851dd44`.

Todos os tickers pedidos no modo offline usam essa mesma série; `requested_ticker`
é contexto, nunca identificação do ativo observado. Os testes exercitam WEGE3,
IBOV e ausência de ticker com yfinance indisponível, além de respostas simuladas
do download ajustado e seus erros. O CI não consulta o Yahoo. CSV e metadados
fazem parte do wheel; a verificação de empacotamento também calibra sem rede.

## Limites de interpretação

O GBM assume parâmetros constantes, incrementos logarítmicos normais independentes
e ausência de saltos. Não modela regimes, caudas pesadas, correlações entre ativos,
rebalanceamento, aportes, custos, impostos, inflação ou câmbio. Os valores são
nominais e a unidade monetária visual é genérica. Uma janela histórica escolhida
não mede o risco futuro; alterações nessa janela podem mudar muito o drift.
Não se deve tratar a animação ou os percentis como recomendação de investimento.
