    function get_available_candidate_moves(solution, candidateArray):
        moves = [
            (point, origin_idx, endpoint_idx)
            for origin_idx, origin in enumerate(solution) if len(origin) > 1
            for endpoint_idx, endpoint in enumerate(solution)
            if origin_idx != endpoint_idx and endpoint_idx in candidateArray[origin_idx]
            for point in origin
        ]
        random.shuffle(moves)
        return moves
    
    function count_group_gain(groups, origin, endpoint):
        distance_loss = np.sum(distance_matrix[groups[origin], element])
        distance_gain = np.sum(distance_matrix[groups[endpoint], element])
    
        return distance_loss - distance_gain
        
    function local_search_move(initial_solution, initial_cost, is_greedy):
        moves = get_available_moves(initial_solution)
        
        if (len(candidateArray) > 0):
            moves = get_available_candidate_moves(initial_solution, candidateArray)
        else:
            moves = get_available_moves(initial_solution)
    
        for move in moves:
            if localCacheDict:
                if move in localCacheDict:
                    if localCacheDict[move] < 0: continue
                else:
                    localCacheDict[move] = count_group_gain(initial_solution, move)
                
            after_move_cost = new_cost(initial_cost, move)
            
            if after_move_cost[0] / after_move_cost[1] < lowest_cost[0] / lowest_cost[1]:
                best_move = current_move
                lowest_cost = after_move_cost

            if is_greedy and best_move:
                break
    
        if not best_move:
            return [False, False]
    
        return [best_move, lowest_cost]

    function createCandidateArray(solution):
        candidateArray= []
        randomDistanceMatrix = []
        
        for group in solution:
            elIndex = random.randint(0, len(group) - 1)
            checkFrom = group[elIndex]
            randomPointSample = random.sample(range(1, 424), 20)

            randomDistanceMatrix = [distance_matrix[checkFrom, checkTo] for checkTo in randomPointSample]
            randomDistanceMatrix, randomPointSample = zip(*sorted(zip(randomDistanceMatrix, randomPointSample)))

            randomPointSample = [x for x in randomPointSample if x not in group]
            randomPointSample = np.array(randomPointSample)[:5]

            randomGroupSample = [
                group_idx 
                for point in randomPointSample
                for group_idx, group in enumerate(initial_solution) if point in group
            ]

            randomGroupSample = list(set(randomGroupSample))
            candidateArray.append(randomGroupSample)

        return candidateArray
        
    function local_search(initial_solution, initial_cost, is_greedy, cache = False, candidate = False):
        cacheDict = {}
        candidateArray = []
        if cache: cacheDict = {'init': 'init'}
        if candidate: candidateArray = createCandidateArray(initial_solution)
       
        if cache: 
            step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, cacheDict, candidateArray)
        else:
            step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, None, candidateArray)
    
        while step:
            solution = apply_move(solution, step)
            cost = new_cost
            if cache: 
                step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, cacheDict, candidateArray)
            else:
                step, new_cost, newCacheDict = local_search_move(solution, distance_matrix, cost, tactic, None, candidateArray)
    
    return solution
        