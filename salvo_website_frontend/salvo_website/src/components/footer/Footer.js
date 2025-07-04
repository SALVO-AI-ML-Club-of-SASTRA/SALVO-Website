import React from "react";
import { Link } from "react-router-dom";
import "./Footer.css";
import logo from "../../assets/logo.png"; 

function Footer() {
    return (
        <footer className="footer">
            <div className="footer-content">
                <div className="footer-details">
                    <img className='footer-logo' src={logo} alt="Salvo Logo" />
                    <h2 className="footer-title">Salvo</h2>
                    <p className="footer-description">

                    </p>
                </div>
                <div className="footer-links">
                    <a href="#"><i className="fab fa-linkedin"></i></a>
                    <a href="#"><i className="fab fa-instagram"></i></a>
                    <a href="#"><i className="fab fa-github"></i></a>
                </div>
                <div className="footer-contact">
                    <div className="footer-help">
                        <h3>Helpful links</h3>
                        <ul>
                            <li>FAQs</li>
                            <li>Support</li>
                            <li>Chat</li>
                        </ul>
                    </div>
                    <div className="footer-contact-info">
                    </div>
                    <div className="footer-qr">
                    </div>
                </div>
            </div>
        </footer>
    );
}

export default Footer;
