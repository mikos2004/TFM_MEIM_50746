# Tabela de tópicos — `01_Analise/spikes/eval/samples/`

## Vista geral

| Ficheiro | Tópico | Domínio | Resumo | Par intra-tópico |
|---|---|---|---|---|
| `txt_00_001.txt` | **00** | Desporto (futebol) | RTP — Antevisão Portugal vs País de Gales; estreia de Jorge Jesus; Ronaldo e o objetivo dos mil golos | `txt_00_002.txt` |
| `txt_00_002.txt` | **00** | Desporto (futebol) | DN — Mesma antevisão, conferência de Jorge Jesus; confirmação de Ronaldo no onze e gestão para os jogos seguintes | `txt_00_001.txt` |
| `txt_01_001.txt` | **01** | Política (educação) | DN — Plenário do Parlamento sobre o arranque do ano letivo; críticas do PCP; resposta do ministro Fernando Alexandre | `txt_01_002.txt` |
| `txt_01_002.txt` | **01** | Política (educação) | Jornal Económico — Mesmo debate; foco nas matrículas fora de prazo e nas trocas de acusações entre PS e Governo | `txt_01_001.txt` |
| `txt_02_001.txt` | **02** | Saúde (endocrinologia) | Relatório clínico — Manuel Ferreira, 58a, diabetes mellitus tipo 2 descompensada (poliúria, perda de peso, visão turva) | `txt_02_002.txt` |
| `txt_02_002.txt` | **02** | Saúde (endocrinologia) | Relatório clínico — Rosa Almeida, 63a, diabetes tipo 2 recentemente diagnosticada (fadiga, sede, infeções urinárias) | `txt_02_001.txt` |
| `txt_03_001.txt` | **03** | Saúde (cardiologia) | Relatório clínico — Joaquim Sousa, 67a, hipertensão essencial não controlada, sem complicações | `txt_03_002.txt` |
| `txt_03_002.txt` | **03** | Saúde (cardiologia) | Relatório clínico — Fernanda Costa, 71a, hipertensão não controlada em contexto de insuficiência cardíaca | `txt_03_001.txt` |
| `txt_04_001.txt` | **04** | Saúde (pneumologia) | Relatório clínico — Tiago Ribeiro, 34a, pneumonia adquirida na comunidade (tosse produtiva, febre, consolidação) | `txt_04_002.txt` |
| `txt_04_002.txt` | **04** | Saúde (pneumologia) | Relatório clínico — Beatriz Lopes, 46a, bronquite aguda em doente asmática (tosse seca, pieira, sibilos) | `txt_04_001.txt` |

## Estrutura por tópico

| Tópico | Nome descritivo | Nº docs | Tipo de intra-semelhança |
|---|---|---|---|
| **00** | Desporto — Seleção Nacional de futebol | 2 | **Muito alta** — ambos cobrem o mesmo evento (antevisão do jogo), partilham nomes próprios (Jorge Jesus, Cristiano Ronaldo, País de Gales, Alvalade), só variam na fonte e no enfoque |
| **01** | Política — Debate parlamentar sobre educação | 2 | **Alta** — mesmo evento, mesmas personagens (Fernando Alexandre, José Luís Carneiro, PCP), mas um artigo é mais longo e descritivo, o outro mais focado nas matrículas |
| **02** | Saúde — Diabetes tipo 2 | 2 | **Alta** — mesma doença, sintomas sobreponíveis (sede, poliúria, cansaço, visão turva), faixas etárias próximas |
| **03** | Saúde — Hipertensão / cardiovascular | 2 | **Média** — mesma patologia de base, mas um caso é HTA simples e o outro tem insuficiência cardíaca associada (sintomas e exames bastante diferentes) |
| **04** | Saúde — Infeções respiratórias | 2 | **Média-baixa** — um é pneumonia bacteriana, outro bronquite asmática; ambos com tosse e febre, mas etiologia, achados e terapêutica divergem |

## Relação

| Par | Relação |
|---|---|
| `txt_00_*` (futebol intra) | mesmo evento |
| `txt_01_*` (política intra) | mesmo evento |
| `txt_02_*` (diabetes intra) | mesma doença |
| `txt_03_*` (HTA intra) | mesma doença, comorbilidades diferentes |
| `txt_04_*` (respiratório intra) | mesma área, diagnósticos diferentes |
| Futebol vs Política | inter-tópico |
| Futebol vs Saúde | inter-tópico |
| Política vs Saúde | inter-tópico |
| Diabetes vs HTA | inter-tópico (ambos saúde) |
| Diabetes vs Pneumologia | inter-tópico (ambos saúde) |
| HTA vs Pneumologia | inter-tópico (ambos saúde) |