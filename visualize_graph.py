# %%
from main import graph

png_bytes = graph.get_graph().draw_mermaid_png()

with open("graph_diagram.png", "wb") as f:
    f.write(png_bytes)

print("Saved graph_diagram.png")

# %%
