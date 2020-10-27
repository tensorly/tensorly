:no-toc:

.. tensorly documentation


.. only:: latex

    TensorLy: Tensor Learning in Python
    ====================================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   user_guide/index
   modules/api
   auto_examples/index
   development_guide/index
   Notebooks <https://github.com/JeanKossaifi/tensorly-notebooks>
   about


.. only:: html

   .. raw:: html

         <section class="hero is-bold">
            <div class="hero-body home-hero">
                <div class="container has-text-centered">

                    <h1 class="title is-1">
                        TensorLy: Simple and Fast Tensor Learning in Python
                    </h1>
                    <hr/>

                    <div class="columns is-vcentered">

                        <div class="column is-8 ">
                            <figure class="image">
                                <img src="_static/TensorLy-pyramid.png" class="logo" alt="TensorLy logo">
                            </figure>
                        </div>

                        <div class="column is-4">
                            <div class="container tensorly-functionalities-container">
                                <div class="content tensorly-functionalities">
                                <ul>
                                    <li> Tested and Optimised </li>
                                    <li> Pure Python </li>
                                    <li> Flexible Backends for: <ul>
                                            <li> <a target="_blank" href="https://numpy.org/"> NumPy </a> </li>
                                            <li> <a target="_blank" href="https://pytorch.org/"> PyTorch </a> </li>
                                            <li> <a target="_blank" href="https://www.tensorflow.org/"> TensorFlow </a> </li>
                                            <li> <a target="_blank" href="https://jax.readthedocs.io/en/latest/"> JAX </a> </li>
                                            <li> <a target="_blank" href="https://mxnet.apache.org/versions/1.7.0/"> Apache MXNet </a> </li>
                                            <li> <a target="_blank" href="https://cupy.dev/"> CuPy </a> </li>
                                        </ul>
                                    </li>
                                    <li> Thorough Documentation </li>
                                    <li> Minimal Dependencies </li>
                                </ul>
                                </div>
                            </div>
                            <p class="has-text-centered">
                            <a class="button is-large is-primary is-outlined" href="installation.html">
                                Get started
                            </a>
                            </p>
                        </div>


                    </div>
                </div>

                <div class="hero-discover container has-text-centered">
                    <p class="title">Discover TensorLy's functionalities!</p>
                </div>
                <div class="container has-text-centered">
                    <br/>
                    <a href="#functionalities">
                    <span>
                        <i class="fa fa-chevron-down" aria-hidden="true"></i>
                    <span>
                    </a>
                </div>

            </div>

            <div class="section home-main">
                <a name="functionalities" style="visibility: hidden;"></a>

                <div class="container">
                    <div class="tile is-ancestor">
                            <div class="tile is-parent">
                            <a class="tile is-child box" href="installation.html">
                                <p class="title">Install TensorLy</p>
                                <p class="subtitle">Installation Instructions </p>
                            </a>
                            </div>

                            <div class="tile is-parent">
                            <a class="tile is-child box" href="user_guide/index.html">
                                <p class="title">User Guide</p>
                                <p class="subtitle">A Friendly Guide to Tensor Learning</p>
                            </a>
                            </div>

                            <div class="tile is-parent">
                            <a class="tile is-child box" href= "auto_examples/index.html">
                                <p class="title">Examples</p>
                                <p class="subtitle">See Usage Examples With Code</p>
                            </a>
                            </div>
                    </div>
                    
                    <div class="tile is-ancestor">
                            <div class="tile is-parent">
                            <a class="tile is-child box" href="modules/api.html">
                                <p class="title">
                                    <i class="fa fa-book" aria-hidden="true"></i>
                                    API
                                </p>
                                <p class="subtitle">Functions and Classes Documentation</p>
                            </a>
                            </div>

                            <div class="tile is-parent">
                            <a class="tile is-child box" href="about.html">
                                <p class="title">About Us</p>
                                <p class="subtitle">About the Developers</p>
                            </a>
                            </div>

                            <div class="tile is-parent">
                            <a class="tile is-child box" href= "https://github.com/tensorly/tensorly">
                                <p class="title">
                                    <span class="icon"><i class="fa fa-github"></i></span>
                                    <span>Contribute</span>
                                </p>
                                <p class="subtitle">Source Code on Github</p>
                            </a>
                            </div>
                    </div>
                </div>

            </div>
        </section>



   .. raw:: html

        <div class="container has-text-centered">

            <script type="text/javascript">
            function disp(s) {
                var win;
                var doc;    
                win = window.open("", "WINDOWID");
                doc = win.document;
                doc.open("text/plain");
                doc.write("<pre>" + s + "</pre>");
                doc.close();
                }                   
            </script>                   
                                                    
                                                
            <script>                
            function toggle_modal(id){
                var modal = document.getElementById(id);
                modal.classList.toggle('is-active');
            };
            </script>     


            <div class="modal" id="tensorly_bibtex">
                <div class="modal-background" onclick="javascrip:toggle_modal('tensorly_bibtex');"></div>
                <div class="modal-content">
                    <div class="box">
                    <div class="content">
                    <code class="language-latex" data-lang="latex">
                    @article{tensorly, <br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;author = {Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic}, <br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;title = {TensorLy: Tensor Learning in Python}, <br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;journal = {Journal of Machine Learning Research (JMLR)} <br/>
                        &nbsp;&nbsp;&nbsp;&nbsp;volume = {20},
                        &nbsp;&nbsp;&nbsp;&nbsp;number = {26},
                        &nbsp;&nbsp;&nbsp;&nbsp;year = {2019}, <br/>
                        } <br/>
                    </code>
                    </div>
                    </div>                                              
                </div>
                <button class="modal-close" onclick="javascrip:toggle_modal('tensorly_bibtex');"></button>
        </div>



        <br/> <br/>

            <div class="card has-text-left">
                <header class="card-header">
                    <p class="card-header-title">
                        If you use TensorLy, please cite:
                    </p>

                    <a class="card-header-icon" onclick="javascrip:toggle_modal('tensorly_bibtex');" >
                        <span class="icon" style="margin-right:0.7em">
                        [bibtex]
                        </span>
                    </a>
                </header>

                <div class="card-content">
                Jean Kossaifi, Yannis Panagakis, Anima Anandkumar and Maja Pantic, <strong> TensorLy: Tensor Learning in Python</strong>, 
                Journal of Machine Learning Research, Year: 2019, Volume: 20, Issue: 26, Pages: 1−6.
                <br/><a href="http://jmlr.org/papers/v20/18-277.html">http://jmlr.org/papers/v20/18-277.html</a>.
                </div>

        </div>
    </div>


