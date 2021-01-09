'''
@Description: Helper functions for all Capstone Project 
'''

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

def bounding_region(actor):
    '''
    #TODO: Given the track_id of an actor in the scene (AV, AGENT, OTHER) generate the physics based 
        bounding box information as discussed w/ Sasha in the presentation slides.
    '''

    # Sample coordinates structure:
    l1 = Point(0, 10) 
    r1 = Point(10, 0) 

    return l1, r1
  
def doOverlap(av_track_id, agent_track_id): 
    '''
    Input: av_l, av_r, ag_l, ag_r <Point class> - The points representing the bounding regions of the AV and Agent
    Output: <bool> True if the AV and Agent's bounding regions overlap, otherwise false.

    Description: Returns true if two rectangles(av_l, av_r) and (ag_l, ag_r) overlap 
    '''

    # Compute physics based bounding region for the AV and the AGENT.
    # NOTE: Based on implementation, the AV bounding region can be computed and cached beforehand.
    av_l, av_r = bounding_region(av_track_id)
    ag_l, ag_r = bounding_region(agent_track_id)
      
    # If one rectangle is on left side of other 
    if(av_l.x >= ag_r.x or ag_l.x >= av_r.x): 
        return False
  
    # If one rectangle is above other 
    if(av_l.y <= ag_r.y or ag_l.y <= av_r.y): 
        return False
  
    return True