/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *	Modified on: July 22, 2018
 *		Author: Saumya Trivedi
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>


#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	if(!is_initialized){
		num_particles = 30; // set number of particles
		
		// define a random engine
		default_random_engine gen;

		
		// creating Gaussian noise at current GPS measurements x, y, theta.
		normal_distribution<double> noise_x(x, std[0]);
		normal_distribution<double> noise_y(y, std[1]);
		normal_distribution<double> noise_theta(theta, std[2]);
		
		for (int i=0; i<num_particles; i++){
			particles[i].id = i;						// setting particle id for each i'th particle
			particles[i].x = noise_x(gen);				// adding noise with mean x of GPS measurement
			particles[i].y = noise_y(gen);				// adding noise with mean y of GPS measurement
			particles[i].theta = noise_theta(gen);		// adding noise with mean theta of GPS measurement
			particles[i].weight = 1.0;					// intializing each weight to 1.0
			
			weights.push_back(particles[i].weight);		// adding each weight to weights vector
		}
		
		is_initialized = true;

	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
		
	

	// define a random engine
	default_random_engine gen;

	for (int i=0; i<num_particles; i++){
		// calculating new precdicted states of particles using 
		// bicycle motion model
		if (abs(yaw_rate) < 0.0001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else{
			particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		
		// adding Gaussian noise with mean as prediction x, y, theta.
		normal_distribution<double> noise_x(particles[i].x, std_pos[0]);
		normal_distribution<double> noise_y(particles[i].y, std_pos[1]);
		normal_distribution<double> noise_theta(particles[i].theta, std_pos[2]);
		
		particles[i].x = noise_x(gen);
		particles[i].y = noise_y(gen);
		particles[i].theta = noise_theta(gen);
		
	}
	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	/*
		Here nearest neighbor method is used to find the closest landmark
		to the observed landmark from the sensor measurement of the vehicle
		at each time step.
	*/
	double curr_dist, low_dist, obs_x, obs_y, map_x, map_y;
	int map_id, closest_landmark_id;
	
	for (unsigned int i=0; i<observations.size(); i++){
		// initializing min distance to really big number
		low_dist = numeric_limits<double>::max();
		// closest landmark id to something not found on map
		closest_landmark_id = -1;
		
		obs_x = observations[i].x;
		obs_y = observations[i].y;
		
		// searching through the list of landmarks to find the closest one to observed one
		for (unsigned int j=0; j<predicted.size(); j++){
			map_x = predicted[j].x;
			map_y = predicted[j].y;
			map_id = predicted[j].id;
			curr_dist = dist(obs_x, obs_y, map_x, map_y);
			
			if(curr_dist < low_dist){
				low_dist = curr_dist;
				closest_landmark_id = map_id;
			}
		}
		observations[i].id = closest_landmark_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	// Define vectors to be used
	vector<LandmarkObs> pred_landmarks;
	vector<LandmarkObs> transformed_obs;
	vector<int> associations;
	vector<double> sense_x;
	vector<double> sense_y;
	
	double sum_weights = 0.0;
	
	for (int i=0; i<num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		
		/*
		1.  Find the landmarks in the sensor range of each particle
			and push them to pred_landmarks vector
		*/
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			double map_x = map_landmarks.landmark_list[j].x_f;
			double map_y = map_landmarks.landmark_list[j].y_f;
			int map_id = map_landmarks.landmark_list[j].id_i;
			
			// filtering the landmarks in the sensor range 
			double L_dist = dist(p_x, p_y, map_x, map_y);
			if (L_dist < sensor_range){
				pred_landmarks.push_back(LandmarkObs{map_id, map_x, map_y});
			}
		}
		
		/*
		2.	Transform observations, from vehicle(particle) co-ordinates to map
			co-ordinates, as we don't know the current position of the vehicle
		*/
		for (unsigned int j=0; j<observations.size(); j++){
			double obs_x = observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta) + p_x;
			double obs_y = observations[j].x * sin(p_theta) + observations[j].y * cos(p_theta) + p_y;
			transformed_obs.push_back(LandmarkObs{observations[j].id, obs_x, obs_y});
		}
		
		/*
		3.	Update transformed observation id's using the nearest neighbor method
		*/
		dataAssociation(pred_landmarks, transformed_obs);
		
		/*
		4.	Based on transformed observations and predicted landmarks nearby,
			we find the weight (Multivariate Gaussian Probability) of each particle
		*/
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double sig_x2 = sig_x * sig_x;
		double sig_y2 = sig_y * sig_y;
		double norm_term = 1.0/(2.0 * M_PI * sig_x * sig_y);
		
		//reinit weight
		particles[i].weight = 1.0;
				
		for (unsigned int j=0; j<transformed_obs.size(); j++){
			
			int Tobs_id = transformed_obs[j].id;
			
			for (unsigned int k=0; k<pred_landmarks.size(); k++){
				int pL_id = pred_landmarks[k].id;
				
				if (Tobs_id == pL_id){
					double Tobs_x = transformed_obs[j].x;
					double Tobs_y = transformed_obs[j].y;
					double pL_x = pred_landmarks[k].x;
					double pL_y = pred_landmarks[k].y;
					
					// set associations for each landmark
					associations.push_back(pL_id);
					sense_x.push_back(Tobs_x);
					sense_y.push_back(Tobs_y);
					
					double prob = norm_term * exp(-1.0 * ((pow((Tobs_x - pL_x), 2)/(2.0 * sig_x2)) + (pow((Tobs_y - pL_y), 2)/(2.0 * sig_y2))));
					particles[i].weight *= prob;
				}
			}
		}
		sum_weights += particles[i].weight;
				
		// set associations for current particle
		SetAssociations(particles[i], associations, sense_x, sense_y);
		
		// clear association data
		associations.clear();
		sense_x.clear();
		sense_y.clear();
	}
	
	// normalizing the weights and adding it to weights vector
	for (unsigned int i=0; i<particles.size(); i++){
		weights[i] = particles[i].weight/sum_weights;
	}
}

void ParticleFilter::resample() {
	// define a random engine
	default_random_engine gen;
	
	/**** Method 1 ****/
	
	/*
	// list all particle indices
	uniform_int_distribution<int> indices(0,num_particles-1);
	
	// select a random indice
	int random_index = indices(gen);
	
	// define beta
	double beta = 0.0;
	
	// find the maximum weight 
	double max_w = *max_element(weights.begin(), weights.end());

	// create a uniform distribution of number between 0 - 2*max_weight
	uniform_real_distribution<double> random_w(0.0, (2.0*max_w));

	for (unsigned int i=0; i< particles.size(); i++){
		// select a random value for beta
		beta += random_w(gen);
		// check if the weight of particle is greater than beta
		while (beta > weights[random_index]){
			beta -= weights[random_index];
			random_index = (random_index + 1) % num_particles;
			resampled_particles.push_back(particles[random_index]);
		}
	} 
	*/
	
	/**** Method 2 ****/
	
	discrete_distribution<> d(weights.begin(), weights.end());

	for (unsigned int i=0; i < weights.size(); ++i)
	{
		resampled_particles[i] = particles[d(gen)];
	}

	particles = resampled_particles;	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
