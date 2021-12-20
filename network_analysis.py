import pandas as pd
import networkx as nx


def network_analysis(df):
    au_df = pd.DataFrame(0, index=df['author'].unique(), columns=df['author'].unique())

    for index, row in df.iterrows():
        t_index = row['parent_id'][3:]
        match = df[df['id'] == t_index]
        try:
            if len(match) > 0:
                au_df.loc[row['author'], match['author'][0]] += 1
        except KeyError:
            continue

    net = nx.from_pandas_edgelist(au_df)
    edges = net.edges()

    betCent = nx.betweenness_centrality(net, normalized=True, endpoints=True)
    node_size = [v * 2000 for v in betCent.values()]
    node_color = [20000.0 * net.degree(v) for v in net]

    pos = nx.circular_layout(net)
    weights = [net[u][v]['weight'] / 20 for u, v in edges]
    options = {
        "edge_color": "gray",
    }
    nx.draw(net, pos, width=weights, node_size=node_size, node_color=node_color, **options)

    # Calculate network measures
    nx.average_clustering(net)
    nx.density(net)
    nx.info(net)

    # To draw as a force directed graph with normalised node sizes based on betweenness centrality:
    pos = nx.spring_layout(net)
    nx.draw(net, pos=pos, node_size=node_size)