# Buraco negro — ray tracing relativístico

Simulação de um buraco negro (Schwarzschild e Kerr) com disco de acreção, feita
traçando a trajetória real da luz no espaço-tempo curvo. Gera imagens PNG e
vídeos MP4 (vertical para Reels/TikTok ou horizontal).

> **Versão em tempo real (GPU, interativa, com malha do espaço-tempo):** abra
> `realtime/index.html` no navegador. Veja [realtime/README.md](realtime/README.md).

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Nada é instalado globalmente. O `ffmpeg` do sistema é usado se existir; se
não, o vídeo é codificado com `imageio-ffmpeg` (com aviso).

## Uso

```bash
# preview rápido (320x180, segundos)
.venv/bin/python -m blackhole render --preset preview
# imagem Full HD (faz benchmark e pede confirmação)
.venv/bin/python -m blackhole render --preset hd
# vídeo vertical 1080x1920, 15 s, 30 fps
.venv/bin/python -m blackhole render --preset video
# vídeo horizontal 1920x1080
.venv/bin/python -m blackhole render --preset video --landscape
# buraco negro girando (Kerr)
.venv/bin/python -m blackhole render --preset preview --spin 0.9
# só estimar o tempo
.venv/bin/python -m blackhole bench --preset video
```

Opções úteis: `--threads N` (padrão: núcleos − 2), `--scale 0.25` (teste em
resolução menor), `--frames 10` (só os primeiros quadros), `--inclination 75`,
`--r-cam 30`, `--fov 60`, `--temp 6500`, `--r-out 20`, `--aa 3`,
`--exposure 0.8`, `--seconds 12`, `--fps 30`, `-y` (pula a confirmação).

Saídas em `outputs/`. Para Kerr, o nome do arquivo inclui o spin
(`preview_a+0.90_...png`).

### Segurança do PC

- CPU limitada a (núcleos − 2) por padrão (`--threads`).
- `hd` e `video` sempre rodam um benchmark curto, mostram o tempo estimado e
  pedem confirmação (a menos que você passe `-y`).
- A imagem é feita em faixas (tiles) gravadas em disco; o vídeo grava cada
  quadro em PNG. Se você interromper (Ctrl+C), rodar o mesmo comando de novo
  **retoma de onde parou**. Uso de RAM fica bem abaixo de 1–2 GB.
- Cada raio tem passo máximo e número máximo de passos (30 000): nada entra
  em loop infinito.
- Barra de progresso com ETA (tqdm).

Referência (12 núcleos, 10 threads): preview ~5 s; vídeo vertical completo
(450 quadros, 1080x1920, AA 3x3) ~6–7 h. Para algo mais rápido:
`--aa 2`, `--scale 0.5` ou `--seconds 10`.

### Céu personalizado

Coloque uma imagem equiretangular (2:1, ex.: um panorama da Via Láctea) em
`./assets/` (png/jpg/exr/hdr). Ela será usada no lugar do céu procedural e
também é distorcida pela lente gravitacional.

## Testes

```bash
.venv/bin/pytest -q
```

Validações automáticas (`tests/test_physics.py`):

| Teste | O que verifica |
|---|---|
| horizonte | r_h = 2M e um raio radial cai em r = 2M |
| esfera de fótons | u = 1/3 (r = 3M) é órbita circular de u'' + u = 3u²; raio com b ≈ b_c gira várias vezes em r ≈ 3M |
| deflexão | integrador Hamiltoniano = equação orbital u'' + u = 3u²; ≈ 4M/b (e 4M/b + 15πM²/4b² até 0,5%) para b = 300, 1000, 3000 |
| sombra | b_c = 3√3 M ≈ 5,196 por bisseção, e pelo tamanho angular visto pela câmera |
| ISCO | r_ISCO = 6M (a=0) e 2,32M (a=0,9) |
| redshift | visto de cima, g = √(1 − 3M/r) (gravitacional √(1−2M/r) × Doppler transversal) |
| Kerr | parâmetros críticos prógrado/retrógrado batem com a fórmula analítica (sombra achatada) |

## Como funciona (técnico, curto)

- **Unidades**: G = c = M = 1. Distâncias em "massas" M (para o Sol, M ≈ 1,5 km).
- **Geodésicas**: formulação Hamiltoniana em coordenadas de Boyer-Lindquist,
  com energia E = 1 e momento angular L conservados, integrada por **RK4 com
  passo adaptativo** (passo pequeno perto do horizonte e da esfera de fótons,
  grande longe). Escolhemos isso em vez de u'' + u = 3u² porque funciona em 3D
  e generaliza direto para Kerr; a equação orbital fica como checagem nos
  testes. Schwarzschild é exatamente o caso a = 0.
- **Câmera**: pinhole num observador ZAMO (que gira junto com o espaço-tempo
  em Kerr). Os raios são traçados **de trás pra frente** (da câmera para a cena).
- **Disco**: fino, no plano equatorial, de r_ISCO até 20M; T(r) ∝ r^(−3/4);
  cada cruzamento do plano soma emissão (opacidade 93%), por isso aparecem
  imagem primária, secundária e anel de fótons.
- **Cor**: lei de Planck integrada nas curvas de cor CIE → RGB; tone mapping
  ACES, bloom e grão leve.
- **Supersampling adaptativo**: um passe com 1 amostra/pixel, depois k×k
  amostras só nos pixels com bordas fortes (sombra, anel de fótons, bordas do
  disco, estrelas).

## A física por trás de cada efeito (roteiro para os vídeos)

### 1. A sombra: o "buraco" preto no meio
Nada escapa de dentro do **horizonte de eventos** (para um buraco negro sem
rotação, raio 2M — o "raio de Schwarzschild"). Mas a mancha preta que você vê é
**maior** que o horizonte: é a **sombra**. Qualquer raio de luz que passe a
menos de 3√3 M ≈ 5,2M do centro (o "parâmetro de impacto crítico") é engolido.
Por isso a sombra aparece ~2,6 vezes maior que o horizonte. Foi exatamente essa
sombra que o Event Horizon Telescope fotografou em M87* e Sagitário A*.

### 2. O anel de fótons: a luz que dá voltas
Em r = 3M existe a **esfera de fótons**: lá a gravidade é tão forte que a luz
pode andar em círculo em volta do buraco negro. Raios que passam bem pertinho
dela dão uma, duas, várias voltas antes de escapar e chegar até nós. Isso
forma o fio de luz fininho e brilhante colado na borda da sombra.

### 3. O disco "por cima" e "por baixo" ao mesmo tempo
Estamos olhando o disco quase de lado, mas vemos a parte de trás dele
**por cima** do buraco negro e também **por baixo**. Isso é a **lente
gravitacional**: a gravidade curva a luz que vem da parte de trás do disco e a
joga por cima (e por baixo) do buraco em direção à câmera. A imagem de baixo,
mais fina, é a **imagem secundária**: luz que deu meia volta a mais.

### 4. Um lado mais brilhante que o outro (Doppler e beaming)
O gás do disco gira em volta do buraco negro a até metade da velocidade da
luz (perto da ISCO). O lado que vem **na nossa direção** fica mais brilhante e
mais azulado; o que **se afasta** fica mais escuro e avermelhado. É o
**efeito Doppler relativístico** somado ao **beaming**: em velocidades
relativísticas a luz é "jogada para frente", como o farol de um carro. A
intensidade escala com g⁴, onde g = (frequência observada)/(frequência
emitida) — uma diferença de g de 1,3 para 0,7 dá ~12× mais brilho de um lado.

### 5. Redshift gravitacional: a luz "cansa" ao sair do poço
A luz emitida perto do buraco negro perde energia para escapar da gravidade:
sua frequência cai pelo fator √(1 − 2M/r). Somado ao fato de que relógios
andando rápido em órbita marcam o tempo mais devagar (dilatação temporal),
a parte interna do disco parece mais fria e mais fraca do que realmente é.
A simulação usa o fator g completo (gravidade + movimento + aberração).

### 6. Por que o disco tem uma borda interna (ISCO)
Abaixo de 6M (para um buraco sem rotação) **não existem órbitas estáveis**:
qualquer gás que chega ali mergulha em espiral no buraco negro. Essa é a ISCO
(órbita circular estável mais interna). Por isso o disco começa ali e não
encosta no horizonte.

### 7. A cor do disco (corpo negro)
O gás é aquecido por atrito enquanto espirala para dentro, e brilha como um
**corpo negro**, igual a um ferro em brasa: mais quente = mais branco-azulado,
mais frio = mais laranja-avermelhado. A temperatura cai para fora como
r^(−3/4) (modelo de Novikov-Thorne simplificado). A cor de cada ponto vem da
lei de Planck, depois de aplicar o desvio g.

### 8. Estrelas distorcidas, duplicadas e o anel de Einstein
A gravidade age como uma lente gigante sobre as estrelas do fundo. Perto da
sombra as estrelas viram arcos, aparecem **duplicadas** (uma de cada lado) e,
quando uma estrela fica exatamente atrás, vira um **anel de Einstein**. Ao
girar a câmera, as estrelas parecem "escorrer" em volta do buraco negro.

### 9. Buraco negro girando (Kerr, `--spin 0.9`)
Buracos negros reais giram. O giro **arrasta o próprio espaço-tempo** (efeito
Lense-Thirring): a luz que vai a favor da rotação consegue chegar mais perto
sem cair, e a que vai contra é engolida mais longe. Resultado: a sombra fica
**achatada de um lado** e deslocada, e a ISCO encolhe (2,32M para a = 0,9),
então o disco chega mais perto e fica muito mais quente e brilhante.

### 10. O tempo no vídeo
O disco tem textura e cada anel gira com a sua própria velocidade kepleriana
(dentro mais rápido que fora), então as manchas viram espirais com o tempo.
Usamos até o atraso da luz: partes do disco cuja luz demora mais para chegar
aparecem "no passado".

## Estrutura

```
src/blackhole/
  geodesic.py   métrica de Kerr, Hamiltoniano, RK4 adaptativo, equação orbital
  tracer.py     kernel Numba: câmera ZAMO, disco, redshift, céu
  color.py      Planck -> RGB
  sky.py        céu procedural / mapa em assets/
  render.py     tiles, supersampling adaptativo, retomada
  post.py       bloom, ACES, grão
  pipeline.py   imagem, vídeo, benchmark, ffmpeg
  config.py     presets e caminho da câmera do vídeo
  __main__.py   CLI
tests/          validações físicas (pytest), incluindo shader vs. offline
realtime/       versão WebGL2 em tempo real (index.html) + validação/benchmark
outputs/        imagens, vídeos, quadros, cache do céu
```

Nota para quem editar o código: o Numba guarda cache compilado por arquivo.
Se você alterar `geodesic.py`, apague `src/blackhole/__pycache__` para que
`tracer.py` seja recompilado com a mudança.
