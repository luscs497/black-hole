# Buraco negro em tempo real (WebGL2)

Versão interativa do simulador: o mesmo ray tracing relativístico de
`src/blackhole`, rodando por pixel na GPU, a 60 fps, com uma **malha do
espaço-tempo** em que você pode mover o buraco negro.

## Como abrir

Dê dois cliques em **`realtime/index.html`** (ou arraste o arquivo para o
Chrome, Edge ou Firefox). É um arquivo só, sem servidor, sem build e sem
dependências externas. Precisa de WebGL2, que qualquer navegador atual tem.

Dica: o Chrome/Edge gravam o vídeo direto em **MP4** (H.264). O Firefox grava
em WebM (para postar, converta com `ffmpeg -i video.webm video.mp4`).

## Controles

| Ação | Como |
|---|---|
| Orbitar a câmera | arrastar com o mouse (ou o dedo) |
| Aproximar / afastar | roda do mouse |
| Inclinação do disco | `↑` `↓` |
| Girar em volta | `←` `→` |
| **Mover o buraco negro pela malha** | `W` `A` `S` `D` (frente/lados, relativos à câmera), `Q` `E` (descer/subir) |
| Voo automático pela malha | `F` |
| Trocar o modo da malha | `G` (desligada → 3D → plano → funil) |
| Doppler + beaming liga/desliga | `1` |
| Redshift gravitacional liga/desliga | `2` |
| Bloom liga/desliga | `3` |
| Disco liga/desliga | `4` |
| Estrelas liga/desliga | `5` |
| Órbita automática da câmera | `Espaço` |
| Qualidade − / + | `[` `]` |
| Gravar / parar (baixa o vídeo) | `R` |
| Screenshot PNG | `P` |
| Fechar / abrir o painel | `H`, o `×` no topo do painel, ou a aba "Controles" na borda direita (a escolha fica salva) |
| Modo cinema (sem interface + órbita automática) | `C` |

No painel também tem: slider de **spin** (0 a 0,99), temperatura do disco, raio
externo, velocidade do tempo, espaçamento/brilho/alcance da malha, FOV,
exposição, qualidade, escala de resolução, intensidade do bloom e **formato de
gravação** (tela cheia, **vertical 1080×1920** para Reels/TikTok, ou horizontal
1920×1080).

**Céu de verdade:** clique em "Carregar céu equiretangular" e escolha uma imagem
de `assets/`, ou arraste a imagem para a janela. O navegador não deixa uma
página aberta via `file://` ler a pasta sozinha, por isso é preciso escolher o
arquivo. A imagem também sofre a lente gravitacional.

## A malha do espaço

- **Funil (Flamm), o padrão**: uma "lona" de linhas abaixo do disco, afundando
  perto do buraco negro como o paraboloide de Flamm (a forma da geometria de
  Schwarzschild "desenhada" em 3D). Com `WASD` o buraco anda e as linhas
  deslizam por baixo dele, como uma bola pesada rolando num tecido.
- **Plano**: a mesma lona, só que plana. Fica mais fácil ver a lente
  gravitacional dobrando as linhas retas.
- **Malha 3D**: um reticulado cúbico de linhas preenchendo o espaço. Perto da
  sombra as linhas se curvam e aparecem duplicadas.

Em todos os modos as linhas são **traçadas de verdade** pelos raios curvos (não
são um desenho por cima). Por isso aparecem dobradas, duplicadas e até atrás do
buraco negro. As linhas também sofrem **redshift gravitacional**: perto do
horizonte ficam mais avermelhadas e apagadas, porque a luz perde energia ao sair
do poço gravitacional.

O funil é uma ilustração didática: a física da luz é a real, mas a lona não
existe de fato no espaço. Mover o buraco negro também é uma simplificação
quase-estática (não modelamos um buraco negro em alta velocidade).

## Física (igual ao render offline)

- Geodésicas nulas de **Kerr** (Schwarzschild = spin 0), com E, L e a constante
  de Carter Q conservados, ZAMO como câmera e RK4 com passo adaptativo (passos
  menores perto do horizonte e da esfera de fótons) e limite de passos por raio.
- Disco fino da ISCO até ~20M, T ∝ r^(−3/4), cor por corpo negro (Planck → CIE
  → RGB, a mesma LUT do offline), fator g completo (gravidade + Doppler) com
  I ∝ g⁴, imagens secundárias e anel de fótons, e textura de gás em rotação
  kepleriana diferencial.
- Estrelas procedurais (hash) com cores de corpo negro, mais a Via Láctea.
  Tudo passa pela lente gravitacional, e o tamanho de cada estrela acompanha a
  ampliação da lente.
- Pós-processamento em passes separados: bloom (pirâmide de downsample e
  upsample), tone mapping filmic ACES, vinheta e grão.

**Diferença técnica em relação ao offline:** a GPU calcula em float32. A forma
Hamiltoniana usada no offline (float64) perde precisão em float32 por
cancelamento catastrófico. Por isso o shader usa as mesmas equações escritas em
"tempo de Mino", com μ = cos θ, e reprojeta a cada passo no vínculo H = 0. A
passagem rente ao eixo de rotação é resolvida analiticamente. Os detalhes estão
nos comentários do shader.

## Performance

- **Escala de resolução dinâmica**: mede o tempo de GPU de cada quadro (timer
  query; sem ele, usa o intervalo entre quadros) e ajusta a resolução interna
  para manter ~60 fps. A imagem é ampliada com filtro Catmull-Rom para continuar
  nítida.
- **Qualidade** (1 a 5): muda o tamanho do passo e o máximo de passos por raio.
- Medido numa RX 580 em 1920×1080: ~34 ms por quadro em resolução cheia
  (qualidade 3), e **60 fps com escala ~0,6**. Rode `realtime/bench.sh` para
  medir na sua máquina.

## Validação

```bash
.venv/bin/python realtime/validate.py      # ou: .venv/bin/pytest -q
```

Roda o shader no Chrome headless (com a GPU real) e compara com o render
offline (Numba, float64):

1. **Sombra**: uma faixa de 2048 px com a câmera a r = 1000M. Os pixels
   capturados batem **pixel a pixel** com o offline, e o raio da sombra dá
   b = 5,1935 (3√3 = 5,1962, diferença menor que 1 pixel). Em Kerr (a = 0,9)
   as bordas prógrada e retrógrada dão 2,846 e 6,831 (analítico: 2,844 e
   6,832).
2. **Direções**: uma grade 32×18 da câmera padrão, para a = 0 e a = 0,9. Nenhum
   pixel muda de estado (capturado/escapou) e o erro de direção fica em
   mediana 0,03° e máximo 0,2°.
3. **Compilação**: os logs de compilação e link de todos os shaders saem vazios
   (sem warnings). Conferido nos backends OpenGL, Vulkan e SwiftShader.

Outras ferramentas de desenvolvimento: `realtime/bench.sh` (tempo por quadro),
`python3 realtime/shot.py saida.png "grid=1&spin=0.9"` (captura do canvas), e
`index.html?bench&stepsmap` (mapa de passos por raio).
