def build_topology(value, topology, visited=set()):
    if value not in visited:
        visited.add(value)
        for child in value.previous:
            build_topology(child, topology, visited)
        topology.append(value)