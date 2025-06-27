import React, { useState } from 'react';
import './Login.css';
import googleIcon from '../../assets/google.svg';
import Nav from '../../components/navigation/Nav';

function Login() {
    const [showSignIn, setShowSignIn] = useState(false);

    const handleShowSignUp = (e) => {
        e.preventDefault();
        setShowSignIn(false);
    };

    const handleShowSignIn = (e) => {
        e.preventDefault();
        setShowSignIn(true);
    };

    return (
        <div className="LoginPage">
            <div className="animated-lines-bg">
                <div className="circuit-element rect1"></div>
                <div className="circuit-element rect2"></div>
                <div className="circuit-element rect3"></div>
                
                <div className="circuit-element node node1"></div>
                <div className="circuit-element node node2"></div>
                <div className="circuit-element node node3"></div>
                <div className="circuit-element node node4"></div>
                
                <div className="circuit-element corner corner1"></div>
                <div className="circuit-element corner corner2"></div>
                
                <div className="circuit-element tjunction tjunction1"></div>
                
                <div className="circuit-element connector connector1"></div>
                <div className="circuit-element connector connector2"></div>
            </div>


            <Nav />
            <h1 className="header">S A L V O</h1>
            <div className="login-container">
                {showSignIn ? (
                    <div id='SignIn' className='sign-in'>
                        <h2>Sign In</h2>
                        <form>
                            <div className='input'>
                                <i className="fa fa-user"></i>
                                <input type="text" id="username" name="username" required placeholder='Username' />
                            </div>
                            <div className='input'>
                                <i className="fa fa-lock"></i>
                                <input type="password" id="password" name="password" required placeholder='Password' />
                            </div>
                            <div className='remember-me'>
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                <label htmlFor="remember-me">Remember me</label>
                            </div>
                            <div className='actions'>
                            <p><a href="#">Forgot your password?</a></p>
                            <p>Don't have an account? <a href="#" onClick={handleShowSignUp}>Sign Up</a></p>
                            </div>
                            <p>OR</p>
                            <button type="button"><img src={googleIcon} alt="Google Icon" />Sign in with Google</button>
                        <button type="submit">Sign Up</button>
                        </form>

                    </div>
                ) : (
                    <div id='SignUp' className='sign-up'>
                        <h2>Sign Up</h2>
                        <form>
                            <div className='input'>
                                <i className="fa fa-user"></i>
                                <input type="text" id="username" name="username" required placeholder='Username' />
                            </div>
                            <div className='input'>
                                <i className="fa fa-envelope"></i>
                                <input type="email" id="email" name="email" required placeholder='Email' />
                            </div>
                            <div className='input'>
                                <i className="fa fa-lock"></i>
                                <input type="password" id="password" name="password" required placeholder='Password' />
                            </div>
                            <p>OR</p>
                            <button type="button"><img src={googleIcon} alt="Google Icon" />Sign in with Google</button>
                        <p>Already have an account? <a href="#" onClick={handleShowSignIn}>Sign In</a></p>
                        <p>By signing up, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.</p>
                        <button type="submit">Sign Up</button>
                        </form>
                    </div>
                )}
            </div>
            <div className='socials-icons'>
                <a href="#"><i className="fab fa-linkedin"></i></a>
                <a href="#"><i className="fab fa-instagram"></i></a>
                <a href="#"><i className="fab fa-github"></i></a>
            </div>
        </div>
    );
}

export default Login;