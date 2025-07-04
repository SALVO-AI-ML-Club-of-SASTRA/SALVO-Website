import React from 'react';
import './Home.css'; 
import Nav from '../../components/navigation/Nav';
import Footer from '../../components/footer/Footer';

function Home() {
    return (
        <div className='home-page'>
            <Nav />
            <div className="home-container">
                <h1 className="home-title">Welcome to Salvo</h1>
                <p className="home-description">
                    Explore the universe of Salvo, where you can find everything from the latest news to community discussions.
                </p>
            </div>
            <Footer />
        </div>
    );
}

export default Home;