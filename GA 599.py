import pygame
import time
import random
import math
import numpy
from pygame import gfxdraw
import plotly.express as px
import pandas as pd
 
def main():
    # ------------------------------- pygame init ------------------------------- #
    pygame.init()
    clock = pygame.time.Clock()
    game_size = 600
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    yellow = (255,255,0)
    fps = 90
    boundary_size = 5
    max_generations = 50
    generation = 1
    timer = 0
    pop_size = 20

    game_display = pygame.display.set_mode((game_size, game_size))
    pygame.display.set_caption('Monkey GA')
    font = pygame.font.Font('freesansbold.ttf', 14)
    
    monkey_img = pygame.image.load('creature.png')
    creature_sprite = pygame.transform.scale(monkey_img, (30, 30))

    # ------------------------------ creature traits ----------------------------- #
    steering_weights = 0.2
    perception_radius_mutation_range = 10
    initial_gather_radius = 5
    # reproduction_rate = 0.0005
    initial_perception_radius = 100
    max_vel = 10
    initial_max_force = 0.5
    # mutation_rate = 0.2
    mutation_proportion = 0.1

    health = 100
    max_food = 25
    max_poison = 10
    nutrition = [20, -50]
 
    # -------------------------------- game memory ------------------------------- #
    creatures = [] 
    food = []
    poison = []
    experiment = []


    def vitality():
        percent_health = creature.health/health
        opacity = int(percent_health*255)
        return(opacity)
 
    def magnitude_calc(vector):
        x = 0
        for i in vector:
            x += i**2
        magnitude = x**0.5
        return(magnitude)
 
    def normalise(vector):
        magnitude = magnitude_calc(vector)
        if magnitude != 0:
            vector = vector/magnitude
        return(vector)

    def plot_results(final):
        # df = px.data.gapminder().query("continent=='Oceania'")
        # df = pd.DataFrame(dict(
        #     Generation = [1, 3, 2, 4],
        #     Change = [1, 2, 3, 4]
        # ))
        fig = px.line(final, x="Generation", y="Percent Change", color='gene', title="Gene Change vs Generation")
        fig.show()
        # print(final)

    def avg_genes(gen):
        avg = [
            ["food attraction", generation, 0],
            ["poison attraction", generation, 0],
            ["food sense", generation, 0],
            ["poison sense", generation, 0],
            ["gather range", generation, 0]
        ]
        for creature in gen:
            # print(creature.dna)
            for i in range(0, len(creature.dna)):
                # print(i, creature.dna[i])
                avg[i][2] += creature.dna[i]
        for i in range(0, len(avg)):
            if len(experiment) is not 0:
                # print(avg)
                # print(experiment[i][2], (avg[i][2]/pop_size))
                avg[i][2] = (((avg[i][2]/pop_size) - experiment[i][2])/experiment[i][2])*100
            else:
                avg[i][2] = (avg[i][2]/pop_size)
        # print(avg)
        return avg


    def generation_turnover(epic):
        
        based = Reproduction(pop_size).replace_pop(creatures)
        
        # calculate = avg_genes(creatures)
        # experiment += calculate
        # print(final)
        if generation is max_generations:
            data = epic
            data[0][2] = 0
            data[1][2] = 0
            data[2][2] = 0
            data[3][2] = 0
            data[4][2] = 0
            final = pd.DataFrame(
                data,
                columns=["gene", "Generation", "Percent Change"]
            )
            plot_results(final)

        return based

    class Reproduction:
        
        def __init__(self, pop_size, mutation_chance = 0.1):
            self.parents = []
            self.pop_size = pop_size
            self.children = []
            self.mutation_chance = mutation_chance
        
        def mutation(self,child_genome):
            mutated_genome = []
            
            for i in range(0, len(child_genome)):
                if random.random() < self.mutation_chance:
                    mutation = random.choice([-1,1])*child_genome[i]*mutation_proportion
                    mutated_genome.append(child_genome[i] + mutation)    
                else:
                    mutated_genome.append(child_genome[i]) 
                    
            return mutated_genome

        def crossover(self,parent_1, parent_2):
            new =[]
            dna1 = parent_1.dna
            dna2 = parent_2.dna
            for i in range(0, len(dna1)):
                new.append(random.choice([dna1[i],dna2[i]]))
            return new
        
        def fitness(self,pop):
            filtered_pop = [x for x in pop if x.alive]
            scored = sorted(filtered_pop, key=lambda creature: creature.score, reverse=False)
            # print("worst: " + str(scored_herd[0].score))
            # print("best: " + str(scored_herd[-1].score))
            return scored
        
        def create_new_creatures(self, parents):
            top_breeders = parents[int(len(parents)/2):]
            creatures = []
            for i in range(0,self.pop_size):
                parent1 = random.choice(top_breeders)
                parent2 = random.choice(top_breeders)
                creature_genome = self.crossover(parent1,parent2)
                creature_genome = self.mutation(creature_genome)
                creature = create_creature(
                    random.uniform(0,game_size-boundary_size),
                    random.uniform(0,game_size-boundary_size),
                    dna=creature_genome
                )
                creatures.append(creature)
            
            return creatures 
        
        def replace_pop(self, parents):
            self.parents = self.fitness(parents)
            self.children = self.create_new_creatures(self.parents)
            
            return self.children

    class create_creature():
        def __init__(self, x, y, dna=False):
            self.position = numpy.array([x,y], dtype='float64')
            self.velocity = numpy.array([random.uniform(-max_vel,max_vel),random.uniform(-max_vel,max_vel)], dtype='float64')
            self.acceleration = numpy.array([0, 0], dtype='float64')

            self.vitality = 255
            self.health = health
            self.score = 0
            self.alive = True

            self.max_vel = 2
            self.max_force = 0.5

            self.size = 5
            self.age = 1
 
            # dna [Vx, Vy, FoodSense, PoisonSense, GatherRadius]
            if dna is False:
                # dna 0 is food attraction, dna 1 is poison attraction
                self.dna = [random.uniform(-initial_max_force/2, initial_max_force), random.uniform(-initial_max_force, initial_max_force/2), 
                random.uniform(0, initial_perception_radius), random.uniform(0, initial_perception_radius), initial_gather_radius]
            else:
                self.dna = dna
 
        def update(self):
            self.velocity += self.acceleration
 
            self.velocity = normalise(self.velocity)*self.max_vel
 
            self.position += self.velocity
            self.acceleration *= 0
            self.health -= 0.2
            self.vitality = vitality()
            self.age += 1
 
        def dead(self):
            if self.health > 0:
                return(False)
            else:
                if self.position[0] < game_size - boundary_size and self.position[0] > boundary_size and self.position[1] < game_size - boundary_size and self.position[1] > boundary_size:
                    food.append(self.position)
                return(True)
 
        def apply_force(self, force):
            self.acceleration += force

        def seek(self, target):
            desired_vel = numpy.add(target, -self.position)
            desired_vel = normalise(desired_vel)*self.max_vel
            steering_force = numpy.add(desired_vel, -self.velocity)
            steering_force = normalise(steering_force)*self.max_force
            return(steering_force)
            #self.apply_force(steering_force)
 
        def eat(self, item_locations, item_type):
            closest = None
            closest_distance = game_size
            creature_x, creature_y = self.position
            item_number = len(item_locations)-1
            
            for i in item_locations[::-1]:
                item_x, item_y = i
                distance = math.hypot(creature_x-item_x, creature_y-item_y)
                # eat if close enough
                if distance < self.dna[4]:
                    item_locations.pop(item_number)
                    new_health = self.health + nutrition[item_type]
                    self.health = min(new_health, 100)
                    self.score += 1
                # find current closest item 
                if distance < closest_distance:
                    closest_distance = distance
                    closest = i
                item_number -=1
            # check if within respective sensing radius and apply forces
            if closest_distance < self.dna[2 + item_type]:
                seek = self.seek(closest) # index)
                seek *= self.dna[item_type]
                seek = normalise(seek)*self.max_force
                self.apply_force(seek)
 
        # ----------------------- Check if creature off screen ----------------------- #
        def boundaries(self):
            x_pos, y_pos = self.position
            if x_pos < 0:
                self.position[0] = game_size - boundary_size
            elif x_pos > game_size:
                self.position[0] = boundary_size
            if y_pos < 0:
                self.position[1] = game_size - boundary_size
            elif y_pos > game_size:
                self.position[1] = boundary_size
            
            # redraw if creature wrapped around screen 
            if x_pos != self.position[0] or y_pos != self.position[1]:
                self.draw_creature()

        # -------------------------- Draw Creature ----------------------------------- #
        def draw_creature(self):
            creature_sprite.set_alpha(self.vitality)
            game_display.blit(creature_sprite, (int(self.position[0]) - 15, int(self.position[1]) - 15))
            pygame.draw.line(game_display, green, (int(self.position[0]), int(self.position[1])), (int(self.position[0] + (self.velocity[0]*self.dna[0]*25)), int(self.position[1] + (self.velocity[1]*self.dna[0]*25))), 3)
            pygame.draw.line(game_display, red, (int(self.position[0]), int(self.position[1])), (int(self.position[0] + (self.velocity[0]*self.dna[1]*25)), int(self.position[1] + (self.velocity[1]*self.dna[1]*25))), 2)
            pygame.draw.circle(game_display, green, (int(self.position[0]), int(self.position[1])), abs(int(self.dna[2])), abs(int(min(2, self.dna[2]))))
            pygame.draw.circle(game_display, red, (int(self.position[0]), int(self.position[1])), abs(int(self.dna[3])), abs(int(min(2, self.dna[3]))))

# ---------------------------------------------------------------------------- #
#                                  Game Setup                                  #
# ---------------------------------------------------------------------------- #
    # number of lads
    for i in range(pop_size):
        creatures.append(create_creature(random.uniform(0,game_size),random.uniform(0,game_size)))


    running = True   
    
    while(running):
        # if generation == 1:
        #     pygame.time.wait(100)

        text = font.render('GENERATION ' + str(generation), True, black)
        textRect = text.get_rect()
        textRect.center = (75, 25)
        game_display.fill(white)
        game_display.blit(text, textRect)
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # chance to generate food and poison
        if random.random()<0.5 and len(food) < max_food:
            food.append(numpy.array([random.uniform(boundary_size, game_size-boundary_size), random.uniform(boundary_size, game_size-boundary_size)], dtype='float64'))
        elif random.random()<0.01 and len(food) > 1:
            food.pop(0)
        if random.random()<0.1 and len(poison) < max_poison:
            poison.append(numpy.array([random.uniform(boundary_size, game_size-boundary_size), random.uniform(boundary_size, game_size-boundary_size)], dtype='float64'))
        elif random.random()<0.01 and len(poison) > 1:
            poison.pop(0)

        for creature in creatures[::-1]:
            if creature.alive is True:
                creature.eat(food, 0)
                creature.eat(poison, 1)
                creature.boundaries()
                creature.update()
                
                creature.draw_creature()

                if creature.dead():
                    creature.alive = False
                    # print("a creature died")
                    # creatures.remove(creature)
 
        for i in food:
            pygame.draw.circle(game_display, green, (int(i[0]), int(i[1])), 3)
        for i in poison:
            pygame.draw.circle(game_display, red, (int(i[0]), int(i[1])), 3)

        timer += 1
        if generation > max_generations:
            running = False

        if timer > 359:
            timer = 0
            # food = []
            # poison = []
            calculate = avg_genes(creatures)
            experiment += calculate
            creatures = generation_turnover(experiment)
            generation += 1

        pygame.display.update()
        clock.tick(fps)
 
    pygame.quit()
    quit()

main()