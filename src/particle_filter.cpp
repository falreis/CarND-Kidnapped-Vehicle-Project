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

	normal_distribution<double> normal_x(x, std[0]);
	normal_distribution<double> normal_y(y, std[1]);
	normal_distribution<double> normal_theta(theta, std[2]);
	default_random_engine rand;

	for(int i=0; i<this->num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = normal_x(rand);
		particle.y = normal_y(rand);
		particle.theta = normal_theta(rand); //NormalizeAngle(normal_theta(rand));
		particle.weight = this->default_weight;
		this->particles.push_back(particle);
		this->weights.push_back(this->default_weight);
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

	double yaw;
	double yaw_dev = yaw_rate * delta_t;

	for(int i=0; i<this->num_particles; i++){
		yaw = this->particles[i].theta;

		//update x, y and theta
		if(yaw_rate < 0.0001){
			this->particles[i].x += velocity * delta_t * cos(yaw);
			this->particles[i].y += velocity * delta_t * sin(yaw);
		}
		else{
			this->particles[i].x += ((velocity/yaw_rate) * (sin(yaw + yaw_dev)-sin(yaw)));
			this->particles[i].y += ((velocity/yaw_rate) * (cos(yaw) - cos(yaw + yaw_dev)));
		}
		this->particles[i].theta += yaw_dev;

		//add random noise
		this->particles[i].x += normal_x(rand);
		this->particles[i].y += normal_y(rand);
		this->particles[i].theta += normal_theta(rand);

		//normalize theta (wrong! why?? normalize angle cause strange behavior and wrong yaw values)
		//this->particles[i].theta = NormalizeAngle(this->particles[i].theta);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int t_size = (observations.size() > predicted.size()) ? predicted.size() : observations.size();

	for(int i=0; i<t_size; i++){
		double min_distance = INFINITY;

		for(int j=0; j<t_size; j++){
			double distance = sqrt(pow((predicted[j].x - observations[i].x),2) + pow((predicted[j].y - observations[i].y),2));
			if(distance < min_distance){
				min_distance = distance;
				observations[i].id = predicted[j].id;
			}
		}
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
		double weight = 1;
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));	//calculate normalization term
		int index_l = -1;

		std::vector<int> associations;
		std::vector<double>sense_x;
		std::vector<double> sense_y;

		//TODO: landmarks_size and observantion shouldn't allways be the same??
		int t_size = (t_landmarks.size() > t_observations.size())? t_observations.size() : t_landmarks.size();

		for(int j=0; j<t_size; j++){
			double x_obs = t_observations.at(j).x;
			double y_obs = t_observations.at(j).y;
			int id_obs = t_observations.at(j).id;

			index_l = -1;
			if(t_landmarks.size() > 0 && t_observations.size() > 0){
				for(int k=0; (k<t_landmarks.size() && index_l < 0); k++){
					if(t_landmarks.at(k).id == t_observations.at(j).id){
						index_l = k;
					}
				}
			}

			//index of (id t_landmarks == id t_observations)
			if(index_l > -1){
				double mu_x = t_landmarks.at(index_l).x;
				double mu_y = t_landmarks.at(index_l).y;

				//calculate exponent
				double exponent= (pow((x_obs-mu_x),2)/(2*pow(sig_x,2))) + (pow((y_obs-mu_y),2)/ (2*pow(sig_y, 2)));

				//calculate weight using normalization terms and exponent
				weight *= gauss_norm * exp(-exponent);

				//set particle associations
				associations.push_back(id_obs);
				sense_x.push_back(x_obs);
				sense_y.push_back(y_obs);
			}
		}
		this->particles[i].weight = weight;
		this->particles[i] = this->SetAssociations(this->particles[i], associations, sense_x, sense_y);
		this->weights[i] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    discrete_distribution<int> distrib(weights.begin(), weights.end());
	default_random_engine rand;
	std::vector<Particle> new_particles;

	for(int i=0; i<this->num_particles; i++){
		int index = distrib(rand);
		new_particles.push_back(this->particles[index]);
	}
	this->particles = new_particles;
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

vector<LandmarkObs> ParticleFilter::transformLandmarks(const Map &map_landmarks, 
		const double sensor_range, const int particle_filter_index)
{
	vector<LandmarkObs> my_landmarks;
	double pred_value = 0;
	double px = this->particles[particle_filter_index].x;
	double py = this->particles[particle_filter_index].y;

	//transform landmarks into unique coordinate system
	for(int j=0; j<map_landmarks.landmark_list.size(); j++){
		double mx = map_landmarks.landmark_list.at(j).x_f;
		double my = map_landmarks.landmark_list.at(j).y_f;
		int mi = map_landmarks.landmark_list.at(j).id_i;

		pred_value = pow((mx-px),2) + pow((my-py),2);

		if (pred_value <= pow(sensor_range,2)){
			LandmarkObs land;
			land.id = mi;
			land.x = mx;
			land.y = my;
			my_landmarks.push_back(land);
		}
	}

	return my_landmarks;
}

vector<LandmarkObs> ParticleFilter::transformObservations(const vector<LandmarkObs> &observations,
		const int particle_filter_index)
{
	vector<LandmarkObs> my_observations;
	Particle p = this->particles[particle_filter_index];

	//transform observations into unique coordinate system
	for(int j=0; j<observations.size(); j++){
		double ox = observations.at(j).x;
		double oy = observations.at(j).y;

		LandmarkObs observ;
		observ.x = (p.x) + (cos(p.theta)*ox) - (sin(p.theta)*oy);	//transform to map x coordinate
		observ.y = (p.y) + (sin(p.theta)*ox) + (cos(p.theta)*oy);	//transform to map y coordinate
		observ.id = observations.at(j).id;
		my_observations.push_back(observ);
	}

	return my_observations;
}