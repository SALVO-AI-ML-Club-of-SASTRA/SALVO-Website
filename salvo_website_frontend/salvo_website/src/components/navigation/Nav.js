import React, { useState, useRef, useEffect } from 'react';
import { Link } from "react-router-dom";
import "./Nav.css";

function Nav() {
    const loggedIn = false;
    const username = '';
    // Replace with actual logic to determine if user is logged in and their username
    // This could be from context, props, or a global state management solution like Redux
    // const { user } = useContext(UserContext);
    // const loggedIn = user ? true : false;
    // const username = user ? user.name : '';
    
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  const toggleDropdown = () => {
    setIsDropdownOpen(open => !open);
  };

  useEffect(() => {
    const handleClickOutside = event => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, []);


    const [isNavOpen, setIsNavOpen] = useState(false);

    const handleNavIconClick = () => {
        setIsNavOpen(open => !open);
    };

    const handleLinkClick = () => {
        setIsNavOpen(false);
    };

    return (
        <div className={`nav${isNavOpen ? ' nav-open' : ''}`}>
            <div className="nav-icon" onClick={handleNavIconClick}>
                <i id='nav-icon' className={`fa ${isNavOpen ? 'fa-times' : 'fa-bars'}`}></i>
            </div>
            <div className={`nav-list`}>
                <div className={`nav-links${isNavOpen ? ' show' : ''}`}>
                    <Link to="/" className="nav-link" onClick={() => setIsNavOpen(false)}>Home</Link>
                    <Link to="/projects" className="nav-link" onClick={() => setIsNavOpen(false)}>Projects</Link>
                    <Link to="/learn" className="nav-link" onClick={() => setIsNavOpen(false)}>Learn</Link>
                    <Link to="/contact" className="nav-link" onClick={() => setIsNavOpen(false)}>Contact</Link>
                </div>
                <div className="nav-profile" onClick={handleLinkClick}>
                    <div className="icon" ref={dropdownRef}>
                        <i className="fa fa-user" onClick={toggleDropdown}></i>
                        <div className={`profile-dropdown ${isDropdownOpen ? '' : 'hidden'}`}>
                            {loggedIn ? (
                                <>
                                    <p>Welcome, {username}!</p>
                                    <Link to="/profile" className="nav-link" onClick={handleLinkClick}>Profile</Link>
                                    <Link to="/settings" className="nav-link" onClick={handleLinkClick}>Settings</Link>
                                    <Link to="/logout" className="nav-link" onClick={handleLinkClick}>Logout</Link>
                                </>
                            ) : (
                                <>
                                    <p>Please log in</p>
                                    <Link to="/login" className="nav-link" onClick={handleLinkClick}>Login</Link>                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Nav;