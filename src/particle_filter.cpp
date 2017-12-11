/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	this->num_particles = 100;
	double default_weight = 1.0f / this->num_particles;

	this->weights.resize(this->num_particles, default_weight);

	normal_distribution<double> normal_x(x, std[0]);
	normal_distribution<double> normal_y(y, std[1]);
	normal_distribution<double> normal_theta(theta, std[2]);
	default_random_engine rand;

	for(int i=0; i<this->num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = normal_x(rand);
		particle.y = normal_y(rand);
		particle.theta = NormalizeAngle(normal_theta(rand));
		particle.weight = default_weight;
		this->particles.push_back(particle);
	}

	this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> normal_x(0, std_pos[0]);
	normal_distribution<double> normal_y(0, std_pos[1]);
	normal_distribution<double> normal_theta(0, std_pos[2]);
	default_random_engine rand;

	double yaw0 = std_pos[2];
	double yaw_dev = yaw_rate * delta_t;

	for(int i=0; i<this->num_particles; i++){
		//update x, y and theta
		this->particles[i].x += ((velocity/yaw_rate) * (sin(yaw0 + yaw_dev)-sin(yaw0)));
		this->particles[i].y += ((velocity/yaw_rate) * (cos(yaw0) - cos(yaw0 + yaw_dev)));
		this->particles[i].theta += yaw_dev;

		//add random noise
		this->particles[i].x += normal_x(rand);
		this->particles[i].y += normal_y(rand);
		this->particles[i].theta += normal_theta(rand);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i=0; i<observations.size(); i++){
		double min_id = 0;
		double min_distance = INFINITY;

		for(int j=0; j<predicted.size(); j++){
			double x = predicted[j].x - observations[i].x;
			double y = predicted[j].y - observations[i].y;
			double distance = sqrt(pow(x,2) + pow(y,2));

			if(distance < min_distance){
				min_distance = distance;
				min_id = predicted[j].id;
			}
		}
		observations[i].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i=0; i<this->num_particles; i++){
		//transform landmarks into unique coordinate system
		vector<LandmarkObs> t_landmarks = this->transformLandmarks(map_landmarks, sensor_range, i);
		
		//transform observations into unique coordinate system
		vector<LandmarkObs> t_observations = this->transformObservations(observations, i);

		//make data association
		this->dataAssociation(t_landmarks, t_observations);

		//define variables
		double weight = 1.0;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];

		std::vector<int> associations;
		std::vector<double>sense_x;
		std::vector<double> sense_y;

		for(int j=0; j<t_observations.size(); j++){
			double x_obs = t_observations.at(j).x;
			double y_obs = t_observations.at(j).y;
			int id_obs = t_observations.at(j).id;

			double mu_x = t_landmarks.at(j).x;
			double mu_y = t_landmarks.at(j).y;
			int mu_id = t_landmarks.at(j).id;	

			//calculate normalization term
			double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

			//calculate exponent
			double exponent= (pow((x_obs - mu_x),2)/(2 * pow(sig_x,2))) + (pow((y_obs - mu_y),2)/ pow((2 * sig_y), 2));

			//calculate weight using normalization terms and exponent
			weight += gauss_norm * exp(-exponent);

			//set particle associations
			associations.push_back(id_obs);
			sense_x.push_back(x_obs);
			sense_y.push_back(y_obs);
		}
		this->particles[i].weight = exp(weight);
		//this->particles[i] = this->SetAssociations(this->particles[i], associations, sense_x, sense_y);

		this->particles[i].associations = associations;
		this->particles[i].sense_x = sense_x;
		this->particles[i].sense_y = sense_y;
		this->weights[i] = this->particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int> distrib(weights.begin(), weights.end());
	default_random_engine rand;
	std::vector<Particle> tmp;

	for(int i=0; i<this->num_particles; i++){
		int index = distrib(rand);
		tmp.push_back(this->particles[index]);
		this->weights[i] = this->particles[index].id;
	}
	this->particles = tmp;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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

vector<LandmarkObs> ParticleFilter::transformLandmarks(const Map &map_landmarks, 
		const double sensor_range, const int particle_filter_index)
{
	vector<LandmarkObs> my_landmarks;
	double pred_value;

	//transform landmarks into unique coordinate system
	for(int j=0; j<map_landmarks.landmark_list.size(); j++){
		double mx = map_landmarks.landmark_list.at(j).x_f;
		double my = map_landmarks.landmark_list.at(j).y_f;
		double mi = map_landmarks.landmark_list.at(j).id_i;

		pred_value = pow((mx - this->particles[particle_filter_index].x), 2) + pow((my - this->particles[particle_filter_index].y), 2);

		if (pred_value <= pow(sensor_range,2)){
			LandmarkObs land;
			land.x = map_landmarks.landmark_list.at(j).x_f;
			land.y = map_landmarks.landmark_list.at(j).y_f;
			land.id = map_landmarks.landmark_list.at(j).id_i;
			my_landmarks.push_back(land);
		}
	}

	return my_landmarks;
}

vector<LandmarkObs> ParticleFilter::transformObservations(const vector<LandmarkObs> &observations,
		const int particle_filter_index)
{
	vector<LandmarkObs> my_observations;
	int pfi = particle_filter_index;

	//transform observations into unique coordinate system
	for(int j=0; j<observations.size(); j++){
		double ox = observations.at(j).x;
		double oy = observations.at(j).y;

		LandmarkObs observ;
		//transform to map x coordinate
		observ.x = (this->particles[pfi].x) + (cos(this->particles[pfi].theta)*ox) - (sin(this->particles[pfi].theta)*oy);

		//transform to map y coordinate
		observ.y = (this->particles[pfi].y) + (sin(this->particles[pfi].theta)*ox) + (cos(this->particles[pfi].theta)*oy);

		observ.id = observations.at(j).id;
		my_observations.push_back(observ);
	}

	return my_observations;
}